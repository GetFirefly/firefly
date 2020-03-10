use core::alloc::{AllocErr, Layout};
use core::ptr::{self, NonNull};

use crate::alloc::realloc_fallback;
use crate::sys::sysconf::MIN_ALIGN;

#[inline]
pub unsafe fn alloc(layout: Layout) -> Result<(NonNull<u8>, usize), AllocErr> {
    let layout_size = layout.size();
    if layout.align() <= MIN_ALIGN && layout.align() <= layout_size {
        NonNull::new(libc::malloc(layout_size) as *mut u8)
            .ok_or(AllocErr)
            .map(|ptr| (ptr, layout_size))
    } else {
        #[cfg(target_os = "macos")]
        {
            if layout.align() > (1 << 31) {
                return Err(AllocErr);
            }
        }
        aligned_alloc(&layout)
    }
}

#[inline]
pub unsafe fn alloc_zeroed(layout: Layout) -> Result<(NonNull<u8>, usize), AllocErr> {
    let layout_size = layout.size();
    if layout.align() <= MIN_ALIGN && layout.align() <= layout_size {
        NonNull::new(libc::calloc(layout_size, 1) as *mut u8)
            .ok_or(AllocErr)
            .map(|ptr| (ptr, layout_size))
    } else {
        let (ptr, ptr_size) = alloc(layout)?;
        ptr::write_bytes(ptr.as_ptr(), 0, layout_size);
        Ok((ptr, ptr_size))
    }
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
        realloc_fallback(ptr, layout, new_size)
    }
}

#[inline]
pub unsafe fn free(ptr: *mut u8, _layout: Layout) {
    libc::free(ptr as *mut libc::c_void)
}

#[inline]
unsafe fn aligned_alloc(layout: &Layout) -> Result<(NonNull<u8>, usize), AllocErr> {
    let mut ptr = ptr::null_mut();
    let layout_size = layout.size();
    let result = libc::posix_memalign(&mut ptr, layout.align(), layout_size);
    if result != 0 {
        return Err(AllocErr);
    }
    Ok((NonNull::new_unchecked(ptr as *mut u8), layout_size))
}
