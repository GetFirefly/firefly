use core::alloc::{AllocError, Layout};
use core::ptr::{self, NonNull};

use crate::alloc::realloc_fallback;
use crate::sys::MIN_ALIGN;

#[inline]
pub fn alloc(layout: Layout) -> Result<MemoryBlock, AllocError> {
    let layout_size = layout.size();
    let ptr = if layout.align() <= MIN_ALIGN && layout.align() <= layout_size {
        libc::malloc(layout_size) as *mut u8
    } else {
        libc::aligned_alloc(layout_size, layout.align()) as *mut u8
    };

    NonNull::new(ptr).ok_or(AllocError).map(|ptr| MemoryBlock {
        ptr,
        size: layout_size,
    })
}

#[inline]
pub fn alloc_zeroed(layout: Layout) -> Result<MemoryBlock, AllocError> {
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

    NonNull::new(ptr).ok_or(AllocError).map(|ptr| MemoryBlock {
        ptr,
        size: layout_size,
    })
}

#[inline]
pub unsafe fn grow(
    ptr: *mut u8,
    layout: Layout,
    new_size: usize,
) -> Result<MemoryBlock, AllocError> {
    let old_size = layout.size();
    let block = self::realloc(ptr, layout, new_size)?;
    AllocInit::init_offset(init, block, old_size);
    Ok(block)
}

#[inline]
pub unsafe fn shrink(
    ptr: *mut u8,
    layout: Layout,
    new_size: usize,
) -> Result<MemoryBlock, AllocError> {
    self::realloc(ptr, layout, new_size)
}

#[inline]
unsafe fn realloc(
    ptr: *mut u8,
    layout: Layout,
    new_size: usize,
) -> Result<MemoryBlock, AllocError> {
    if layout.align() <= MIN_ALIGN && layout.align() <= new_size {
        NonNull::new(libc::realloc(ptr as *mut libc::c_void, new_size) as *mut u8)
            .ok_or(AllocError)
            .map(|ptr| MemoryBlock {
                ptr,
                size: new_size,
            })
    } else {
        realloc_fallback(self, ptr, layout, new_size)
    }
}

#[inline]
pub unsafe fn dealloc(ptr: *mut u8, _layout: Layout) {
    libc::free(ptr as *mut libc::c_void)
}
