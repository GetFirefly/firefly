use core::ptr::{self, NonNull};

use crate::alloc::realloc_fallback;
use crate::alloc::{AllocErr, GlobalAlloc, Layout};
use crate::sys::sysconf::MIN_ALIGN;

#[inline]
pub unsafe fn alloc(layout: Layout) -> Result<(NonNull<u8>, usize), AllocErr> {
    let layout_size = layout.size();
    let ptr = if layout.align() <= MIN_ALIGN && layout.align() <= layout_size {
        libc::malloc(layout_size) as *mut u8
    } else {
        libc::aligned_alloc(layout_size, layout.align()) as *mut u8
    };

    NonNull::new(ptr)
        .ok_or(AllocErr)
        .map(|ptr| (ptr, layout_size))
}

#[inline]
pub unsafe fn alloc_zeroed(layout: Layout) -> Result<NonNull<u8>, AllocErr> {
    let layout_size = layout.size();
    let ptr = if layout.align() <= MIN_ALIGN && layout.align() <= layout_size {
        libc::calloc(layout.size(), 1) as *mut u8
    } else {
        let ptr = libc::aligned_alloc(layout_size, layout.align()) as *mut u8;
        if !ptr.is_null() {
            ptr::write_bytes(ptr, 0, layout_size);
        }
        ptr
    };

    NonNull::new(ptr)
        .ok_or(AllocErr)
        .map(|ptr| (ptr, layout_size))
}

#[inline]
pub unsafe fn realloc(
    ptr: *mut u8,
    layout: Layout,
    new_size: usize,
) -> Result<(NonNull<u8>, usize), AllocErr> {
    if layout.align() <= MIN_ALIGN && layout.align() <= new_size {
        NonNull::new(libc::realloc(ptr as *mut libc::c_void, new_size) as *mut u8)
            .ok_or(AllocErr)
            .map(|ptr| (ptr, new_size))
    } else {
        realloc_fallback(self, ptr, layout, new_size)
    }
}

#[inline]
pub unsafe fn dealloc(ptr: *mut u8, _layout: Layout) {
    libc::free(ptr as *mut libc::c_void)
}
