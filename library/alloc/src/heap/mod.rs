mod generational;
mod semispace;

pub use self::generational::{Generation, GenerationalHeap};
pub use self::semispace::SemispaceHeap;

use alloc::alloc::{AllocError, Allocator, Layout};
use core::cell::UnsafeCell;
use core::ops::Range;
use core::ptr::{self, NonNull};

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
///
pub trait Heap: Allocator {
    /// Returns the address of the first addressable byte of the heap.
    ///
    /// The pointer returned by `heap_start` is always lower than `heap_end`
    fn heap_start(&self) -> *mut u8;

    /// Returns the address of the first addressable byte of the heap which is free for allocation.
    ///
    /// On a new heap, this will return the same address as `heap_start`.
    /// If a heap was perfectly filled, this would return the same address as `heap_end`.
    /// On a heap with at least one allocation, this will return the address of the first byte following
    /// the last allocation.
    fn heap_top(&self) -> *mut u8;

    /// Resets `heap_top` to the given pointer.
    ///
    /// This is extremely dangerous to use without carefully considering what is on the heap when called:
    ///
    /// * You must ensure that `ptr` is within the allocation defined by this heap
    /// * You must ensure that all of the terms on the heap in the region covered by `layout` are properly disposed of,
    /// * You must ensure that `layout` starts at a valid term and ends at a valid term.
    unsafe fn reset_heap_top(&self, ptr: *mut u8);

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
    fn as_ptr_range(&self) -> Range<*mut u8> {
        self.heap_start()..self.heap_end()
    }

    /// Returns a `Range` representing `heap_start..heap_top`
    #[inline]
    fn used_range(&self) -> Range<*mut u8> {
        self.heap_start()..self.heap_top()
    }

    /// Returns the location on this heap where a garbage collection cycle last stopped, if this heap
    /// is generational, and at least one collection has occurred.
    ///
    /// Defaults to `None`.
    ///
    /// Implementations should override this function if they are generational, and should return a
    /// pointer value that is somewhere in the range `heap_start..heap_end` and divides the heap into
    /// two regions: mature and immature allocations. In other words, any allocation which is reachable
    /// during a collection, and is below the high water mark, is considered mature, as it has survived
    /// at least one collection already.
    #[inline]
    fn high_water_mark(&self) -> Option<NonNull<u8>> {
        None
    }

    /// Returns a `Range` representing the portion of the heap which is under the high water mark.
    ///
    /// Unless overridden by an implementation, if the high water mark is `None`, this returns an empty range.
    #[inline]
    fn mature_range(&self) -> Range<*mut u8> {
        let heap_start = self.heap_start();
        heap_start
            ..(self
                .high_water_mark()
                .map(|p| p.as_ptr())
                .unwrap_or(heap_start))
    }

    /// Returns a `Range` representing the portion of the heap which is equal to or higher than the
    /// high water mark.
    ///
    /// Unless overridden by an implementation, if the high water mark is `None`, this is equivalent to `as_ptr_range`.
    #[inline]
    fn immature_range(&self) -> Range<*mut u8> {
        (self.heap_start())
            ..(self
                .high_water_mark()
                .map(|p| p.as_ptr())
                .unwrap_or_else(|| self.heap_end()))
    }

    /// Returns the total size in bytes of the addressable heap
    ///
    /// NOTE: The allocation backing this heap may actually be larger than this size, it only reflects
    /// how large the heap appears to be to consumers of this trait.
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

    /// Returns true if this heap should be garbage collected
    #[inline]
    fn should_collect(&self, gc_threshold: f64) -> bool {
        let used = self.heap_used();
        let size = self.heap_size();
        let threshold = (size as f64 * gc_threshold).ceil() as usize;
        used >= threshold
    }

    /// Returns true if this heap contains the allocation pointed to by `ptr`
    #[inline]
    fn contains(&self, ptr: *const ()) -> bool {
        self.as_ptr_range().contains(&ptr.cast_mut().cast())
    }
}

/// An extension trait for `Heap` which provides additional functionality
/// when a mutable reference is available.
pub trait HeapMut {
    /// Sets the high water mark of the current heap to the given location.
    ///
    /// By default this does nothing.
    ///
    /// # SAFETY
    ///
    /// The pointer given must be on the current heap, and must be equal to or
    /// under `heap_top`.
    #[inline]
    fn set_high_water_mark(&mut self, _ptr: *mut u8) {}
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
    unsafe fn reset_heap_top(&self, ptr: *mut u8) {
        (**self).reset_heap_top(ptr)
    }

    #[inline]
    fn heap_end(&self) -> *mut u8 {
        (**self).heap_end()
    }

    #[inline]
    fn as_ptr_range(&self) -> Range<*mut u8> {
        (**self).as_ptr_range()
    }

    #[inline]
    fn used_range(&self) -> Range<*mut u8> {
        (**self).used_range()
    }

    #[inline]
    fn high_water_mark(&self) -> Option<NonNull<u8>> {
        (**self).high_water_mark()
    }

    #[inline]
    fn mature_range(&self) -> Range<*mut u8> {
        (**self).mature_range()
    }

    #[inline]
    fn immature_range(&self) -> Range<*mut u8> {
        (**self).immature_range()
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
    fn contains(&self, ptr: *const ()) -> bool {
        (**self).contains(ptr)
    }
}

impl<H> HeapMut for &mut H
where
    H: ?Sized + HeapMut,
{
    #[inline]
    fn set_high_water_mark(&mut self, ptr: *mut u8) {
        (**self).set_high_water_mark(ptr)
    }
}

/// This is a simple no-op `Heap` which is always empty and cannot be allocated into.
///
/// Useful for testing, or for expressing emptiness in the context of heap containment.
pub struct EmptyHeap;
impl Heap for EmptyHeap {
    /// Returns the address of the first addressable byte of the heap.
    ///
    /// The pointer returned by `heap_start` is always lower than `heap_end`
    fn heap_start(&self) -> *mut u8 {
        ptr::null_mut()
    }

    #[inline]
    fn heap_top(&self) -> *mut u8 {
        ptr::null_mut()
    }

    unsafe fn reset_heap_top(&self, _ptr: *mut u8) {}

    #[inline]
    fn heap_end(&self) -> *mut u8 {
        ptr::null_mut()
    }

    #[inline(always)]
    fn contains(&self, _ptr: *const ()) -> bool {
        false
    }
}
impl HeapMut for EmptyHeap {}
unsafe impl Allocator for EmptyHeap {
    fn allocate(&self, _layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        Err(AllocError)
    }

    unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: Layout) {}

    unsafe fn grow(
        &self,
        _ptr: NonNull<u8>,
        _old_layout: Layout,
        _new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        Err(AllocError)
    }

    unsafe fn grow_zeroed(
        &self,
        _ptr: NonNull<u8>,
        _old_layout: Layout,
        _new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        Err(AllocError)
    }

    unsafe fn shrink(
        &self,
        _ptr: NonNull<u8>,
        _old_layout: Layout,
        _new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        Err(AllocError)
    }
}

/// This is a simple `Heap` which allocates in a fixed-size buffer
pub struct FixedSizeHeap<const N: usize> {
    top: UnsafeCell<usize>,
    buffer: [u8; N],
}
impl<const N: usize> FixedSizeHeap<N> {
    #[inline(always)]
    fn top(&self) -> usize {
        unsafe { *self.top.get() }
    }
}
impl<const N: usize> Default for FixedSizeHeap<N> {
    fn default() -> Self {
        Self {
            top: UnsafeCell::new(0),
            buffer: [0; N],
        }
    }
}
impl<const N: usize> Heap for FixedSizeHeap<N> {
    #[inline]
    fn as_ptr_range(&self) -> Range<*mut u8> {
        let range = self.buffer.as_slice().as_ptr_range();
        range.start.cast_mut()..range.end.cast_mut()
    }

    /// Returns the address of the first addressable byte of the heap.
    ///
    /// The pointer returned by `heap_start` is always lower than `heap_end`
    fn heap_start(&self) -> *mut u8 {
        self.buffer.as_slice().as_ptr().cast_mut()
    }

    #[inline]
    fn heap_top(&self) -> *mut u8 {
        let bottom = self.as_ptr_range().start;
        unsafe { bottom.add(self.top()) }
    }

    #[inline]
    unsafe fn reset_heap_top(&self, ptr: *mut u8) {
        assert!(self.as_ptr_range().contains(&ptr));
        assert!(self.heap_top() >= ptr);
        let diff = self.heap_top().sub_ptr(ptr);
        let top = self.top.get();
        *top -= diff;
    }

    #[inline]
    fn heap_end(&self) -> *mut u8 {
        self.as_ptr_range().end
    }

    #[inline(always)]
    fn heap_size(&self) -> usize {
        N
    }

    #[inline(always)]
    fn contains(&self, ptr: *const ()) -> bool {
        self.buffer.as_ptr_range().contains(&ptr.cast())
    }
}
impl<const N: usize> HeapMut for FixedSizeHeap<N> {}
unsafe impl<const N: usize> Allocator for FixedSizeHeap<N> {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let size = layout.size();
        if size > self.heap_available() {
            return Err(AllocError);
        }

        let base_ptr = unsafe { NonNull::new_unchecked(self.heap_top()) };
        unsafe {
            *self.top.get() += size;
        }
        Ok(NonNull::slice_from_raw_parts(base_ptr, size))
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        let size = layout.size();
        let base = ptr.as_ptr();
        assert!(self.as_ptr_range().contains(&base));
        if base.add(size) == self.heap_top() {
            *self.top.get() -= size;
        }
    }

    unsafe fn grow(
        &self,
        _ptr: NonNull<u8>,
        _old_layout: Layout,
        _new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        Err(AllocError)
    }

    unsafe fn grow_zeroed(
        &self,
        _ptr: NonNull<u8>,
        _old_layout: Layout,
        _new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        Err(AllocError)
    }

    unsafe fn shrink(
        &self,
        _ptr: NonNull<u8>,
        _old_layout: Layout,
        _new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        Err(AllocError)
    }
}
