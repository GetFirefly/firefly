use core::alloc::{Alloc, AllocErr, Layout};
use core::cmp;
use core::ptr::{self, NonNull};

use crate::sys;

#[derive(Debug, Copy, Clone)]
pub struct SysAlloc;

unsafe impl Alloc for SysAlloc {
    #[inline]
    unsafe fn alloc(&mut self, layout: Layout) -> Result<NonNull<u8>, AllocErr> {
        sys::alloc(layout)
    }

    #[inline]
    unsafe fn alloc_zeroed(&mut self, layout: Layout) -> Result<NonNull<u8>, AllocErr> {
        sys::alloc_zeroed(layout)
    }

    #[inline]
    unsafe fn dealloc(&mut self, ptr: NonNull<u8>, layout: Layout) {
        sys::free(ptr.as_ptr(), layout)
    }

    #[inline]
    unsafe fn realloc(
        &mut self,
        ptr: NonNull<u8>,
        layout: Layout,
        new_size: usize,
    ) -> Result<NonNull<u8>, AllocErr> {
        sys::realloc(ptr.as_ptr(), layout, new_size)
    }
}

pub unsafe fn realloc_fallback(
    ptr: *mut u8,
    old_layout: Layout,
    new_size: usize,
) -> Result<NonNull<u8>, AllocErr> {
    let new_layout = Layout::from_size_align_unchecked(new_size, old_layout.align());
    let new_ptr = sys::alloc(new_layout)?;
    let size = cmp::min(old_layout.size(), new_size);
    ptr::copy_nonoverlapping(ptr, new_ptr.as_ptr(), size);
    sys::free(ptr, old_layout);

    Ok(new_ptr)
}
