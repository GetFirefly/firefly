use core::ptr::{self, NonNull};

use crate::alloc::{prelude::*, realloc_fallback};
use crate::sys::sysconf::MIN_ALIGN;

#[inline]
pub fn allocate(layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
    let layout_size = layout.size();
    if layout.align() <= MIN_ALIGN && layout.align() <= layout_size {
        NonNull::new(unsafe { libc::malloc(layout_size) as *mut u8 })
            .ok_or(AllocError)
            .map(|ptr| NonNull::slice_from_raw_parts(ptr, layout_size))
    } else {
        #[cfg(target_os = "macos")]
        {
            if layout.align() > (1 << 31) {
                return Err(AllocError);
            }
        }
        unsafe { aligned_alloc(&layout) }
    }
}

#[inline]
pub fn allocate_zeroed(layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
    let layout_size = layout.size();
    if layout.align() <= MIN_ALIGN && layout.align() <= layout_size {
        NonNull::new(unsafe { libc::calloc(layout_size, 1) as *mut u8 })
            .ok_or(AllocError)
            .map(|ptr| NonNull::slice_from_raw_parts(ptr, layout_size))
    } else {
        let ptr = allocate(layout)?;
        unsafe {
            ptr::write_bytes(ptr.as_mut_ptr(), 0, layout_size);
        }
        Ok(ptr)
    }
}

#[inline]
pub unsafe fn deallocate(ptr: NonNull<u8>, _layout: Layout) {
    libc::free(ptr.as_ptr() as *mut libc::c_void)
}

#[inline]
pub unsafe fn grow(
    ptr: NonNull<u8>,
    old_layout: Layout,
    new_layout: Layout,
) -> Result<NonNull<[u8]>, AllocError> {
    reallocate(ptr, old_layout, new_layout)
}

#[inline]
pub unsafe fn grow_zeroed(
    ptr: NonNull<u8>,
    old_layout: Layout,
    new_layout: Layout,
) -> Result<NonNull<[u8]>, AllocError> {
    reallocate(ptr, old_layout, new_layout).map(|ptr| {
        let old_size = old_layout.size();
        let dst = ptr.as_mut_ptr().add(old_size);
        let count = new_layout.size() - old_size;
        ptr::write_bytes(dst, 0, count);

        ptr
    })
}

#[inline]
pub unsafe fn shrink(
    ptr: NonNull<u8>,
    old_layout: Layout,
    new_layout: Layout,
) -> Result<NonNull<[u8]>, AllocError> {
    reallocate(ptr, old_layout, new_layout)
}

unsafe fn reallocate(
    ptr: NonNull<u8>,
    old_layout: Layout,
    new_layout: Layout,
) -> Result<NonNull<[u8]>, AllocError> {
    // TODO handle changed align
    assert!(old_layout.align() == new_layout.align());
    let new_size = new_layout.size();

    if new_layout.align() <= MIN_ALIGN && new_layout.align() <= new_size {
        NonNull::new(libc::realloc(ptr.as_ptr() as *mut libc::c_void, new_size) as *mut u8)
            .ok_or(AllocError)
            .map(|ptr| NonNull::slice_from_raw_parts(ptr, new_size))
    } else {
        realloc_fallback(ptr, old_layout, new_layout)
    }
}

#[inline]
unsafe fn aligned_alloc(layout: &Layout) -> Result<NonNull<[u8]>, AllocError> {
    let mut ptr = ptr::null_mut();
    let layout_size = layout.size();
    let result = libc::posix_memalign(&mut ptr, layout.align(), layout_size);
    if result != 0 {
        return Err(AllocError);
    }
    Ok(NonNull::slice_from_raw_parts(
        NonNull::new_unchecked(ptr as *mut u8),
        layout_size,
    ))
}
