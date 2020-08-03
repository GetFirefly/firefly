use core::alloc::Layout;
use core::cmp;
use core::mem;
use core::ops::DerefMut;
use core::ptr::NonNull;

use liblumen_core::sys::sysconf::MIN_ALIGN;
use liblumen_core::util::pointer::{distance_absolute, in_area};

use crate::erts::exception::AllocResult;
use crate::erts::term::prelude::Term;

/// The core trait for allocating on a heap
pub trait HeapAlloc {
    /// Perform a heap allocation.
    ///
    /// If space on the process heap is not immediately available, then the allocation
    /// will be pushed into a heap fragment which will then be later moved on to the
    /// process heap during garbage collection
    unsafe fn alloc(&mut self, need: usize) -> AllocResult<NonNull<Term>> {
        let align = cmp::max(mem::align_of::<Term>(), MIN_ALIGN);
        let size = need * mem::size_of::<Term>();
        let layout = Layout::from_size_align(size, align).unwrap();
        self.alloc_layout(layout)
    }

    /// Same as `alloc`, but takes a `Layout` rather than the size in words
    unsafe fn alloc_layout(&mut self, layout: Layout) -> AllocResult<NonNull<Term>>;
}

impl<T, H> HeapAlloc for T
where
    H: HeapAlloc,
    T: DerefMut<Target = H>,
{
    #[inline]
    unsafe fn alloc(&mut self, need: usize) -> AllocResult<NonNull<Term>> {
        self.deref_mut().alloc(need)
    }

    #[inline]
    unsafe fn alloc_layout(&mut self, layout: Layout) -> AllocResult<NonNull<Term>> {
        self.deref_mut().alloc_layout(layout)
    }
}

/// The base trait for heap implementations
///
/// Provides access to metadata about a heap above and beyond the low-level allocation functions
pub trait Heap: HeapAlloc {
    fn is_corrupted(&self) -> bool;

    /// Returns the lowest address that is part of the underlying heaps' range
    fn heap_start(&self) -> *mut Term;

    /// Returns the address immediately following the most recent
    /// allocation in the underlying heap; it represents the position
    /// at which the next allocation will begin, not accounting for
    /// padding
    fn heap_top(&self) -> *mut Term;

    /// Returns the highest address that is part of the underlying heaps' range
    fn heap_end(&self) -> *mut Term;

    /// Returns the location on this heap where a collection cycle last stopped
    ///
    /// Defaults to `heap_start`, and only requires implementation if this heap
    /// supports distinguishing between mature and immature allocations
    #[inline]
    fn high_water_mark(&self) -> *mut Term {
        self.heap_start()
    }

    /// Returns the total allocated size of the underlying heap
    #[inline]
    fn heap_size(&self) -> usize {
        distance_absolute(self.heap_end(), self.heap_start())
    }

    /// Returns the total number of bytes that are used by
    /// allocations in the underling heap
    #[inline]
    fn heap_used(&self) -> usize {
        distance_absolute(self.heap_top(), self.heap_start())
    }

    /// Returns the total number of bytes that are available for
    /// allocations in the underling heap
    #[inline]
    fn heap_available(&self) -> usize {
        distance_absolute(self.heap_end(), self.heap_top())
    }

    /// Returns true if the underlying heap contains the address represented by `ptr`
    #[inline]
    fn contains<T: ?Sized>(&self, ptr: *const T) -> bool {
        in_area(ptr, self.heap_start(), self.heap_end())
    }

    /// An alias for `contains` which better expresses intent in some places
    ///
    /// Returns true if the given pointer is owned by this process/heap
    #[inline(always)]
    fn is_owner<T: ?Sized>(&self, ptr: *const T) -> bool {
        self.contains(ptr)
    }

    #[cfg(debug_assertions)]
    #[inline]
    fn sanity_check(&self) {
        let hb = self.heap_start();
        let he = self.heap_end();
        let size = self.heap_size() * mem::size_of::<Term>();
        assert_eq!(size, (he as usize - hb as usize), "mismatch between heap size and the actual distance between the start of the heap and end of the stack");
        let ht = self.heap_top();
        assert!(
            hb <= ht,
            "bottom of the heap must be a lower address than or equal to the top of the heap"
        );
    }

    #[cfg(not(debug_assertions))]
    #[inline]
    fn sanity_check(&self) {}
}

impl<T, H> Heap for T
where
    H: Heap,
    T: DerefMut<Target = H>,
{
    fn is_corrupted(&self) -> bool {
        self.deref().is_corrupted()
    }

    #[inline]
    fn heap_start(&self) -> *mut Term {
        self.deref().heap_start()
    }

    #[inline]
    fn heap_top(&self) -> *mut Term {
        self.deref().heap_top()
    }

    #[inline]
    fn heap_end(&self) -> *mut Term {
        self.deref().heap_end()
    }

    #[inline]
    fn high_water_mark(&self) -> *mut Term {
        self.deref().high_water_mark()
    }

    #[inline]
    fn heap_size(&self) -> usize {
        self.deref().heap_size()
    }

    #[inline]
    fn heap_used(&self) -> usize {
        self.deref().heap_used()
    }

    #[inline]
    fn heap_available(&self) -> usize {
        self.deref().heap_available()
    }

    #[inline]
    fn contains<U: ?Sized>(&self, ptr: *const U) -> bool {
        self.deref().contains(ptr)
    }

    #[inline(always)]
    fn is_owner<U: ?Sized>(&self, ptr: *const U) -> bool {
        self.deref().is_owner(ptr)
    }
}
