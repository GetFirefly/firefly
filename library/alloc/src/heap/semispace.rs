use alloc::alloc::{AllocError, Allocator, Layout};
use core::mem;
use core::ops::Range;
use core::ptr::NonNull;

use super::{GenerationalHeap, Heap, HeapMut};

/// This struct implements a semi-space generational heap, like that used in the
/// BEAM for Erlang processes.
///
/// A semi-space heap is characterized by the fact that it uses two heaps to store
/// two different classes of allocations - immature and mature - with all new allocations
/// going into the immature heap. The mature heap is used to hold allocations which have
/// survived at least one garbage collection, and are thus likely to survive subsequent
/// collections as well. As a result, the garbage collector can ignore objects on the mature
/// heap during minor collections, making those cycles much more efficient.
///
/// # Invariants
///
/// An invariant enforced by this semi-space heap is that there are never any pointers
/// from an older generation to a newer generation, i.e. from the mature heap to the immature
/// heap. Any referenced objects on the immature heap by objects being moved to the mature
/// heap must also be moved.
///
/// # Lifecycle
///
/// A semi-space heap goes through a specific set of phases:
///
/// 1. During this initial phase, only the immature heap has been allocated, while an empty
/// mature heap acts as a placeholder until the first GC cycle. All new allocations go to the
/// immature heap.
///
/// 2. The first minor GC cycle occurs. During this cycle, a new immature heap is allocated,
/// and live values reachable from the root set will be moved to the new heap. A high water mark
/// will be set, indicating where the heap was at when the cycle ended. The old immature heap is freed.
///
/// 3. The second minor GC cycle occurs. During this cycle, a new immature heap is allocated, but
/// additionally, a mature heap is allocated with enough space to hold the old immature heap, replacing
/// the initial empty mature heap. Objects in the mature region of the immature heap (i.e. allocations
/// below the high-water mark) are moved to the mature heap. Objects above the high water mark are moved
/// to the new immature heap.
///
/// 4. During this phase, some number of minor GC cycles will occur, until the mature heap fills up and
/// needs to grow. When this happens, a major GC cycle occurs. During a major GC cycle, a new mature heap
/// is allocated and live matured objects from both heaps are swept into the new mature heap. The old mature
/// heap becomes the new immature heap, and the old immature heap is freed.
#[derive(Debug)]
pub struct SemispaceHeap<A, B> {
    immature: A,
    mature: B,
}
impl<A, B> SemispaceHeap<A, B>
where
    A: Heap,
    B: Heap,
{
    /// Creates a new semi-space heap composed of the given immature and mature heaps
    pub fn new(immature: A, mature: B) -> Self {
        Self { immature, mature }
    }
}
unsafe impl<A, B> Allocator for SemispaceHeap<A, B>
where
    A: Heap,
    B: Heap,
{
    #[inline]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.immature.allocate(layout)
    }

    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        if self.mature.contains(ptr.as_ptr().cast()) {
            self.mature.deallocate(ptr, layout);
        } else {
            self.immature.deallocate(ptr, layout);
        }
    }

    #[inline]
    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        if self.mature.contains(ptr.as_ptr().cast()) {
            self.mature.grow(ptr, old_layout, new_layout)
        } else {
            self.immature.grow(ptr, old_layout, new_layout)
        }
    }

    #[inline]
    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        if self.mature.contains(ptr.as_ptr().cast()) {
            self.mature.grow_zeroed(ptr, old_layout, new_layout)
        } else {
            self.immature.grow_zeroed(ptr, old_layout, new_layout)
        }
    }

    #[inline]
    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        if self.mature.contains(ptr.as_ptr().cast()) {
            self.mature.shrink(ptr, old_layout, new_layout)
        } else {
            self.immature.shrink(ptr, old_layout, new_layout)
        }
    }
}
impl<A, B> Heap for SemispaceHeap<A, B>
where
    A: Heap,
    B: Heap,
{
    #[inline]
    fn heap_start(&self) -> *mut u8 {
        self.immature.heap_start()
    }

    #[inline]
    fn heap_top(&self) -> *mut u8 {
        self.immature.heap_top()
    }

    #[inline]
    unsafe fn reset_heap_top(&self, top: *mut u8) {
        self.immature.reset_heap_top(top);
    }

    #[inline]
    fn heap_end(&self) -> *mut u8 {
        self.immature.heap_end()
    }

    #[inline]
    fn as_ptr_range(&self) -> Range<*mut u8> {
        self.immature.as_ptr_range()
    }

    #[inline]
    fn used_range(&self) -> Range<*mut u8> {
        self.immature.used_range()
    }

    #[inline]
    fn high_water_mark(&self) -> Option<NonNull<u8>> {
        self.immature.high_water_mark()
    }

    #[inline]
    fn mature_range(&self) -> Range<*mut u8> {
        self.immature.mature_range()
    }

    #[inline]
    fn immature_range(&self) -> Range<*mut u8> {
        self.immature.immature_range()
    }

    #[inline]
    fn heap_size(&self) -> usize {
        self.immature.heap_size()
    }

    #[inline]
    fn heap_used(&self) -> usize {
        self.immature.heap_used()
    }

    #[inline]
    fn heap_available(&self) -> usize {
        self.immature.heap_available()
    }

    #[inline]
    fn should_collect(&self, gc_threshold: f64) -> bool {
        self.immature.should_collect(gc_threshold)
    }

    #[inline]
    fn contains(&self, ptr: *const ()) -> bool {
        self.immature.contains(ptr) || self.mature.contains(ptr)
    }
}
impl<A, B> HeapMut for SemispaceHeap<A, B>
where
    A: HeapMut,
    B: HeapMut,
{
    #[inline]
    fn set_high_water_mark(&mut self, ptr: *mut u8) {
        self.immature.set_high_water_mark(ptr);
    }
}
impl<A, B> GenerationalHeap for SemispaceHeap<A, B>
where
    A: Heap,
    B: Heap,
{
    type Immature = A;
    type Mature = B;

    #[inline]
    fn is_immature(&self, ptr: *const u8) -> bool {
        self.immature.contains(ptr.cast())
    }

    #[inline]
    fn is_mature(&self, ptr: *const u8) -> bool {
        self.mature.contains(ptr.cast())
    }

    #[inline]
    fn swap_immature(&mut self, new_heap: Self::Immature) -> Self::Immature {
        mem::replace(&mut self.immature, new_heap)
    }

    #[inline]
    fn swap_mature(&mut self, new_heap: Self::Mature) -> Self::Mature {
        mem::replace(&mut self.mature, new_heap)
    }

    #[inline]
    fn immature(&self) -> &Self::Immature {
        &self.immature
    }

    #[inline]
    fn immature_mut(&mut self) -> &mut Self::Immature {
        &mut self.immature
    }

    #[inline]
    fn mature(&self) -> &Self::Mature {
        &self.mature
    }

    #[inline]
    fn mature_mut(&mut self) -> &mut Self::Mature {
        &mut self.mature
    }
}
