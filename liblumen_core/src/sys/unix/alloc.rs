use core::ptr::{self, NonNull};

use crate::alloc::{prelude::*, realloc_fallback};
use crate::sys::sysconf::MIN_ALIGN;

#[inline]
pub fn alloc(layout: Layout) -> Result<MemoryBlock, AllocErr> {
    let layout_size = layout.size();
    if layout.align() <= MIN_ALIGN && layout.align() <= layout_size {
        NonNull::new(unsafe { libc::malloc(layout_size) as *mut u8 })
            .ok_or(AllocErr)
            .map(|ptr| MemoryBlock {
                ptr,
                size: layout_size,
            })
    } else {
        #[cfg(target_os = "macos")]
        {
            if layout.align() > (1 << 31) {
                return Err(AllocErr);
            }
        }
        unsafe { aligned_alloc(&layout) }
    }
}

#[inline]
pub fn alloc_zeroed(layout: Layout) -> Result<MemoryBlock, AllocErr> {
    let layout_size = layout.size();
    if layout.align() <= MIN_ALIGN && layout.align() <= layout_size {
        NonNull::new(unsafe { libc::calloc(layout_size, 1) as *mut u8 })
            .ok_or(AllocErr)
            .map(|ptr| MemoryBlock {
                ptr,
                size: layout_size,
            })
    } else {
        let block = alloc(layout)?;
        unsafe {
            ptr::write_bytes(block.ptr.as_ptr(), 0, layout_size);
        }
        Ok(block)
    }
}
#[inline]
pub unsafe fn grow(
    ptr: *mut u8,
    layout: Layout,
    new_size: usize,
    placement: ReallocPlacement,
    init: AllocInit,
) -> Result<MemoryBlock, AllocErr> {
    if placement != ReallocPlacement::MayMove {
        // We can't guarantee the allocation won't move
        return Err(AllocErr);
    }
    let old_size = layout.size();
    if layout.align() <= MIN_ALIGN && layout.align() <= new_size {
        NonNull::new(libc::realloc(ptr as *mut libc::c_void, new_size) as *mut u8)
            .ok_or(AllocErr)
            .map(|ptr| MemoryBlock {
                ptr,
                size: new_size,
            })
    } else {
        realloc_fallback(ptr, layout, new_size)
    }
}

#[inline]
pub unsafe fn shrink(
    ptr: *mut u8,
    layout: Layout,
    new_size: usize,
    placement: ReallocPlacement,
) -> Result<MemoryBlock, AllocErr> {
    if placement != ReallocPlacement::MayMove {
        // We can't guarantee the allocation won't move
        return Err(AllocErr);
    }
    let old_size = layout.size();
    if layout.align() <= MIN_ALIGN && layout.align() <= new_size {
        NonNull::new(libc::realloc(ptr as *mut libc::c_void, new_size) as *mut u8)
            .ok_or(AllocErr)
            .map(|ptr| MemoryBlock {
                ptr,
                size: new_size,
            })
    } else {
        realloc_fallback(ptr, layout, new_size)
    }
}

#[inline]
pub unsafe fn free(ptr: *mut u8, _layout: Layout) {
    libc::free(ptr as *mut libc::c_void)
}

#[inline]
unsafe fn aligned_alloc(layout: &Layout) -> Result<MemoryBlock, AllocErr> {
    let mut ptr = ptr::null_mut();
    let layout_size = layout.size();
    let result = libc::posix_memalign(&mut ptr, layout.align(), layout_size);
    if result != 0 {
        return Err(AllocErr);
    }
    Ok(MemoryBlock {
        ptr: NonNull::new_unchecked(ptr as *mut u8),
        size: layout_size,
    })
}
