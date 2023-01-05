mod generational;
mod semispace;

pub use self::generational::GenerationalHeap;
pub use self::semispace::SemispaceHeap;

use alloc::alloc::Allocator;
use core::ops::Range;
use core::ptr::NonNull;

/// A `Heap` is a specialized allocator over a fixed-size region of memory.
/// Heaps are growable/shrinkable, can be garbage-collected, and are allocated/freed
/// as a region, rather than in terms of individual allocations.
///
/// Previously, we supported traversing heaps directly, but this doesn't play well with custom
/// types which may have specific alignment requirements that we can't know a priori. Instead,
/// traversing a heap requires a set of roots (i.e. pointers to values on the heap). From those
/// roots, we can trace references of those objects, determine whether or not they are allocated
/// in a specific heap, and from there, decide what to do with them. See `GcBox` for the container
/// type we use to allocate on `Heap` implementations, and how tracing works.
pub trait Heap: Allocator {
    /// Returns the address of the first addressable byte of the heap.
    ///
    /// The pointer returned by `heap_start` is always lower than `heap_end`
    fn heap_start(&self) -> *mut u8;

    /// Returns the address of the first addressable byte of the heap which is free for allocation.
    ///
    /// On a new heap, this will return the same address as `heap_start`.
    /// If a heap was perfectly filled, this would return the same address as `heap_end`.
    /// On a heap with at least one allocation, this will return the address of the first byte
    /// following the last allocation.
    fn heap_top(&self) -> *mut u8;

    /// Returns the address `heap_end` in the exclusive range `heap_start..heap_end`, i.e. it is
    /// one byte past the addressable range of this heap.
    ///
    /// For example, if you have a a heap that starts at address 0, and is 1024B in size, then
    /// this would return the address 1025.
    ///
    /// The pointer returned by this function should _never_ be dereferenced, it should only be used
    /// in cases where performing pointer math is useful
    fn heap_end(&self) -> *mut u8;

    /// Returns a `Range` representing `heap_start..heap_end`
    #[inline]
    fn as_ptr_range(&self) -> Range<*const u8> {
        (self.heap_start() as *const u8)..(self.heap_end() as *const u8)
    }

    /// Returns the location on this heap where a garbage collection cycle last stopped, if this
    /// heap is generational, and at least one collection has occurred.
    ///
    /// Defaults to `None`.
    ///
    /// Implementations should override this function if they are generational, and should return a
    /// pointer value that is somewhere in the range `heap_start..heap_end` and divides the heap
    /// into two regions: mature and immature allocations. In other words, any allocation below
    /// the high water mark is considered a mature allocation that has survived at least one
    /// collection cycle, and is thus more likely to survive subsequent collections.
    #[inline]
    fn high_water_mark(&self) -> Option<NonNull<u8>> {
        None
    }

    /// Returns the total size in bytes of the addressable heap
    ///
    /// NOTE: The allocation backing this heap may actually be larger than this size, it only
    /// reflects how large the heap appears to be to consumers of this trait.
    #[inline]
    fn heap_size(&self) -> usize {
        unsafe { self.heap_end().offset_from(self.heap_start()) as usize }
    }

    /// Returns the total size in bytes of allocated memory on this heap
    #[inline]
    fn heap_used(&self) -> usize {
        unsafe { self.heap_top().offset_from(self.heap_start()) as usize }
    }

    /// Returns the total size in bytes of unallocated memory on this heap
    #[inline]
    fn heap_available(&self) -> usize {
        unsafe { self.heap_end().offset_from(self.heap_top()) as usize }
    }

    /// Returns true if this heap contains the allocation pointed to by `ptr`
    #[inline]
    fn contains<T: ?Sized>(&self, ptr: *const T) -> bool {
        self.as_ptr_range().contains(&ptr.cast())
    }
}

impl<H> Heap for &H
where
    H: ?Sized + Heap,
{
    #[inline]
    fn heap_start(&self) -> *mut u8 {
        (**self).heap_start()
    }

    #[inline]
    fn heap_top(&self) -> *mut u8 {
        (**self).heap_top()
    }

    #[inline]
    fn heap_end(&self) -> *mut u8 {
        (**self).heap_end()
    }

    #[inline]
    fn as_ptr_range(&self) -> Range<*const u8> {
        (**self).as_ptr_range()
    }

    #[inline]
    fn high_water_mark(&self) -> Option<NonNull<u8>> {
        (**self).high_water_mark()
    }

    #[inline]
    fn heap_size(&self) -> usize {
        (**self).heap_size()
    }

    #[inline]
    fn heap_used(&self) -> usize {
        (**self).heap_used()
    }

    #[inline]
    fn heap_available(&self) -> usize {
        (**self).heap_available()
    }

    #[inline]
    fn contains<T: ?Sized>(&self, ptr: *const T) -> bool {
        (**self).contains(ptr)
    }
}
