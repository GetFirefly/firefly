use core::alloc::Layout;
use core::mem;
use core::ptr::NonNull;

use crate::erts::exception::AllocResult;
use crate::erts::term::prelude::{Boxed, ProcBin, Term};

use super::*;

/// A supertrait for generational heaps which uses one heap type for
/// its young generations, and another for its old generations
///
/// `GenerationalHeap` itself does not implement `Heap`, although it
/// could, but instead it is intended for users of a generational heap
/// to request access to a generation and invoke the standard `Heap` API
/// from there
pub trait GenerationalHeap: Heap + VirtualAlloc {
    type Young: Heap + VirtualAlloc;
    type Old: Heap + VirtualAlloc;

    /// Returns true if the given pointer is in the mature region of this heap
    fn is_mature<P: ?Sized>(&self, ptr: *const P) -> bool;

    /// Replaces the current young generation with a new heap
    fn swap_young(&mut self, new_young: Self::Young) -> Self::Young;

    /// Replaces the current old generation with a new heap
    fn swap_old(&mut self, new_old: Self::Old) -> Self::Old;

    /// Get immutable access to the young heap
    fn young_generation(&self) -> &Self::Young;

    /// Get mutable access to the young heap
    fn young_generation_mut(&mut self) -> &mut Self::Young;

    /// Get immutable access to the old heap
    fn old_generation(&self) -> &Self::Old;

    /// Get mutable access to the old heap
    fn old_generation_mut(&mut self) -> &mut Self::Old;
}

/// The standard implementation of a generational heap used by Erlang processes.
///
/// This heap goes through a specific set of phases:
///
///
/// 1.) The initial young heap of some minimum size is allocated, along with an empty
/// old heap; this forms the two spaces of the semi-space heap itself. Until the first
/// collection occurs, all values are allocated on the young heap.
/// 2.) The first minor collection occurs. A new young heap is allocated, and live values
/// reachable from the root set will be moved into the new heap. A high water mark will
/// then be set where that collection ended.
/// 3.) The second minor collection occurs. An old heap is allocated, sized to the usage of
/// the young heap, and replaces the empty heap. Objects in the mature region (the area from
/// the start of the heap to the high water mark) will then be moved to the old heap, rather
/// than to the new young heap. Immature objects are moved to the young generation as usual.
/// When objects are moved to the old heap, a sweep of the old heap is performed as well to
/// ensure that any references to the young generation are also promoted to the old heap. This
/// is done to enforce the property that references are always to objects of an older generation,
/// or the same generation, but never backwards to a younger generation.
/// 4.) A minor collection occurs, if the old heap needs resizing, a full sweep is performed
/// instead; this moves all objects into a fresh young heap. Otherwise, this phase is the
/// same as #3
/// 5.) During a full sweep, the old generation is replaced with an empty heap, just like during
/// initialization, resetting the lifecycle
#[derive(Debug)]
pub struct SemispaceHeap<A, B>
where
    A: Heap + VirtualAlloc,
    B: Heap + VirtualAlloc,
{
    young: A,
    old: B,
}
impl<A, B> SemispaceHeap<A, B>
where
    A: Heap + VirtualAlloc,
    B: Heap + VirtualAlloc,
{
    pub fn new(young: A, old: B) -> Self {
        Self { young, old }
    }

    // Check if either the young generation, or the virtual heap, require
    // collection by comparing usage against a percentage threshold
    #[inline]
    pub fn should_collect(&self, gc_threshold: f64) -> bool {
        // First, check young generation
        let used = self.young.heap_used();
        let unused = self.young.heap_available();
        let threshold = ((used + unused) as f64 * gc_threshold).ceil() as usize;
        if used >= threshold {
            return true;
        }
        // Next, check virtual heap
        let used = self.young.virtual_heap_used();
        let unused = self.young.virtual_heap_unused();
        if unused > 0 {
            let threshold = ((used + unused) as f64 * gc_threshold).ceil() as usize;
            used >= threshold
        } else {
            // We've exceeded the virtual heap size
            true
        }
    }
}
impl<A, B> GenerationalHeap for SemispaceHeap<A, B>
where
    A: Heap + VirtualAlloc,
    B: Heap + VirtualAlloc,
{
    type Young = A;
    type Old = B;

    fn is_mature<P: ?Sized>(&self, ptr: *const P) -> bool {
        use liblumen_core::util::pointer::in_area;
        in_area(ptr, self.young.heap_start(), self.young.high_water_mark())
    }

    fn swap_young(&mut self, new_young: Self::Young) -> Self::Young {
        mem::replace(&mut self.young, new_young)
    }

    fn swap_old(&mut self, new_old: Self::Old) -> Self::Old {
        mem::replace(&mut self.old, new_old)
    }

    #[inline]
    fn young_generation(&self) -> &Self::Young {
        &self.young
    }

    #[inline]
    fn young_generation_mut(&mut self) -> &mut Self::Young {
        &mut self.young
    }

    #[inline]
    fn old_generation(&self) -> &Self::Old {
        &self.old
    }

    #[inline]
    fn old_generation_mut(&mut self) -> &mut Self::Old {
        &mut self.old
    }
}
impl<A, B> Heap for SemispaceHeap<A, B>
where
    A: Heap + VirtualAlloc,
    B: Heap + VirtualAlloc,
{
    #[inline]
    fn heap_start(&self) -> *mut Term {
        self.young.heap_start()
    }

    #[inline]
    fn heap_top(&self) -> *mut Term {
        self.young.heap_top()
    }

    #[inline]
    fn heap_end(&self) -> *mut Term {
        self.young.heap_end()
    }

    #[inline]
    fn heap_size(&self) -> usize {
        self.young.heap_size()
    }

    #[inline]
    fn heap_used(&self) -> usize {
        self.young.heap_used()
    }

    #[inline]
    fn heap_available(&self) -> usize {
        self.young.heap_available()
    }

    #[inline]
    fn contains<T: ?Sized>(&self, ptr: *const T) -> bool {
        self.young.contains(ptr) || self.old.contains(ptr)
    }

    #[inline]
    fn is_owner<T: ?Sized>(&self, ptr: *const T) -> bool {
        self.contains(ptr) || self.virtual_contains(ptr)
    }

    #[inline]
    fn sanity_check(&self) {
        self.young.sanity_check();
        self.old.sanity_check();
    }
}
impl<A, B> HeapAlloc for SemispaceHeap<A, B>
where
    A: Heap + VirtualAlloc,
    B: Heap + VirtualAlloc,
{
    #[inline]
    unsafe fn alloc_layout(&mut self, layout: Layout) -> AllocResult<NonNull<Term>> {
        self.young.alloc_layout(layout)
    }
}
impl<A, B> StackAlloc for SemispaceHeap<A, B>
where
    A: Heap + VirtualAlloc + StackAlloc,
    B: Heap + VirtualAlloc,
{
    #[inline]
    unsafe fn alloca(&mut self, need: usize) -> AllocResult<NonNull<Term>> {
        self.young.alloca(need)
    }

    #[inline]
    unsafe fn alloca_unchecked(&mut self, need: usize) -> NonNull<Term> {
        self.young.alloca_unchecked(need)
    }
}
impl<A, B> StackPrimitives for SemispaceHeap<A, B>
where
    A: Heap + VirtualAlloc + StackPrimitives,
    B: Heap + VirtualAlloc,
{
    #[inline]
    fn stack_size(&self) -> usize {
        self.young.stack_size()
    }

    #[inline]
    unsafe fn set_stack_size(&mut self, size: usize) {
        self.young.set_stack_size(size);
    }

    #[inline]
    fn stack_pointer(&mut self) -> *mut Term {
        self.young.stack_pointer()
    }

    #[inline]
    unsafe fn set_stack_pointer(&mut self, sp: *mut Term) {
        self.young.set_stack_pointer(sp);
    }

    #[inline]
    fn stack_used(&self) -> usize {
        self.young.stack_used()
    }

    #[inline]
    fn stack_available(&self) -> usize {
        self.young.stack_available()
    }

    #[inline]
    fn stack_slot(&mut self, n: usize) -> Option<Term> {
        self.young.stack_slot(n)
    }

    #[inline]
    fn stack_popn(&mut self, n: usize) {
        self.young.stack_popn(n);
    }
}
impl<A, B> VirtualHeap<ProcBin> for SemispaceHeap<A, B>
where
    A: Heap + VirtualAlloc,
    B: Heap + VirtualAlloc,
{
    #[inline]
    fn virtual_size(&self) -> usize {
        self.young.virtual_size() + self.old.virtual_size()
    }

    #[inline]
    fn virtual_heap_used(&self) -> usize {
        self.young.virtual_heap_used() + self.old.virtual_heap_used()
    }

    #[inline]
    fn virtual_heap_unused(&self) -> usize {
        self.young.virtual_heap_unused() + self.old.virtual_heap_unused()
    }
}
impl<A, B> VirtualAllocator<ProcBin> for SemispaceHeap<A, B>
where
    A: Heap + VirtualAlloc,
    B: Heap + VirtualAlloc,
{
    #[inline]
    fn virtual_alloc(&mut self, value: Boxed<ProcBin>) {
        self.young.virtual_alloc(value)
    }

    #[inline]
    fn virtual_free(&mut self, value: Boxed<ProcBin>) {
        let ptr = value.as_ptr();
        assert!(
            self.young.virtual_contains(ptr),
            "can't free term not linked to this virtual heap"
        );
        self.young.virtual_free(value);
    }

    #[inline]
    fn virtual_unlink(&mut self, value: Boxed<ProcBin>) {
        let ptr = value.as_ptr();
        assert!(
            self.young.virtual_contains(ptr),
            "can't unlink term not linked to this virtual heap"
        );
        self.young.virtual_unlink(value);
    }

    #[inline]
    fn virtual_pop(&mut self, value: Boxed<ProcBin>) -> ProcBin {
        let ptr = value.as_ptr();
        assert!(
            self.young.virtual_contains(ptr),
            "can't pop binary not linked to this virtual heap"
        );
        self.young.virtual_pop(value)
    }

    #[inline]
    fn virtual_contains<P: ?Sized>(&self, value: *const P) -> bool {
        self.young.virtual_contains(value) || self.old.virtual_contains(value)
    }

    #[inline]
    unsafe fn virtual_clear(&mut self) {
        self.young.virtual_clear();
        self.old.virtual_clear();
    }
}
