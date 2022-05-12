use alloc::alloc::{Allocator, Global, AllocError, Layout};
use core::ptr::{self, NonNull};
use core::mem;

use crate::term::Term;

pub struct ProcessHeap {
    range: *mut [u8],
    top: *mut u8,
}
impl ProcessHeap {
    const DEFAULT_HEAP_SIZE: 4 * 1024;

    pub fn new() -> Self {
        let layout = Layout::from_size_align(Self::DEFAULT_HEAP_SIZE, mem::align_of::<Term>());
        let nonnull = Global.allocate(layout).unwrap();
        Self {
            range: nonnull.as_ptr(),
            top: nonnull.as_non_null_ptr().as_ptr(),
        }
    }
}
impl Drop for ProcessHeap {
    fn drop(&mut self) {
        let size = ptr::metadata(self.range);
        let layout = Layout::from_size_align(size, mem::align_of::<Term>());
        unsafe {
            Global.deallocate(NonNull::new_unchecked(self.range.cast()), layout)
        }
    }
}
unsafe impl Allocator for ProcessHeap {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let layout = layout.pad_to_align();
        let size = layout.size();

        // Calculate the base pointer of the allocation at the desired alignment,
        // then offset that pointer by the desired size to give us the new top
        let top = self.top;
        let offset = top.align_offset(layout.align());
        let base = unsafe { top.add(offset) };
        let new_top = unsafe { base.add(size) };

        // Make sure the requested allocation fits within the fragment
        let start = self.range.as_ptr();
        let heap_size = self.range.len();
        let range = start..(unsafe { start.add(heap_size) })
        if range.contains(&new_top) {
            Ok(unsafe { NonNull::new_unchecked(ptr::from_raw_parts_mut(base.cast(), size)) })
        } else {
            Err(AllocError)
        }
    }

    unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: Layout) {}
    unsafe fn grow(&self, _ptr: NonNull<u8>, _old_layout: Layout, _new_layout: Layout) -> Result<NonNull<[u8]>, AllocError> { Err(AllocError) }
    unsafe fn grow_zeroed(&self, _ptr: NonNull<u8>, _old_layout: Layout, _new_layout: Layout) -> Result<NonNull<[u8]>, AllocError> { Err(AllocError) }
    unsafe fn shrink(&self, ptr: NonNull<u8>, _old_layout: Layout, new_layout: Layout) -> Result<NonNull<[u8]>, AllocError> { Ok(NonNull::slice_from_raw_parts(ptr, new_layout.size())) }
}
impl Heap for ProcessHeap {
    #[inline]
    fn heap_start(&self) -> *mut u8 {
        self.range.as_mut_ptr()
    }

    #[inline]
    fn heap_top(&self) -> *mut u8 {
        self.top
    }

    #[inline]
    fn heap_end(&self) -> *mut u8 {
        self.heap_start().add(self.range.len())
    }
}
