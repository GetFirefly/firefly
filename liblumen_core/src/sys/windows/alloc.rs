use core::alloc::prelude::*;
use core::ptr::NonNull;

use winapi::shared::minwindef::{DWORD, LPVOID};
use winapi::um::errhandlingapi::GetLastError;
use winapi::um::heapapi::{GetProcessHeap, HeapAlloc, HeapFree, HeapReAlloc};
use winapi::um::winnt::HEAP_ZERO_MEMORY;

use crate::alloc::realloc_fallback;
use crate::sys::sysconf::MIN_ALIGN;

#[repr(transparent)]
struct Header(*mut u8);

#[inline]
pub fn alloc(layout: Layout) -> Result<MemoryBlock, AllocErr> {
    let layout_size = layout.size();
    NonNull::new(alloc_with_flags(layout, 0))
        .ok_or(AllocErr)
        .map(|ptr| MemoryBlock {
            ptr,
            size: layout_size,
        })
}

#[inline]
pub fn alloc_zeroed(layout: Layout) -> Result<MemoryBlock, AllocErr> {
    NonNull::new(alloc_with_flags(layout, HEAP_ZERO_MEMORY))
        .ok_or(AllocErr)
        .map(|ptr| MemoryBlock {
            ptr,
            size: layout_size,
        })
}

#[inline]
pub unsafe fn grow(
    ptr: *mut u8,
    layout: Layout,
    new_size: usize,
    placement: ReallocPlacement,
    init: AllocInit,
) -> Result<MemoryBlock, AllocErr> {
    let old_size = layout.size();
    let block = self::realloc(ptr, layout, new_size, placement)?;
    AllocInit::init_offset(init, block, old_size);
    Ok(block)
}

#[inline]
pub unsafe fn shrink(
    ptr: *mut u8,
    layout: Layout,
    new_size: usize,
    placement: ReallocPlacement,
) -> Result<MemoryBlock, AllocErr> {
    self::realloc(ptr, layout, new_size, placement)
}

#[inline]
unsafe fn realloc(
    ptr: *mut u8,
    layout: Layout,
    new_size: usize,
    placement: ReallocPlacement,
) -> Result<(NonNull<u8>, usize), AllocErr> {
    if placement != ReallocPlacement::MayMove {
        return Err(AllocErr);
    }

    if layout.align() <= MIN_ALIGN {
        NonNull::new(HeapReAlloc(GetProcessHeap(), 0, ptr as LPVOID, new_size) as *mut u8)
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
pub unsafe fn free(ptr: *mut u8, layout: Layout) {
    if layout.align() <= MIN_ALIGN {
        let err = HeapFree(GetProcessHeap(), 0, ptr as LPVOID);
        debug_assert!(err != 0, "Failed to free heap memory: {}", GetLastError());
    } else {
        let header = get_header(ptr);
        let err = HeapFree(GetProcessHeap(), 0, header.0 as LPVOID);
        debug_assert!(err != 0, "Failed to free heap memory: {}", GetLastError());
    }
}

#[inline]
unsafe fn alloc_with_flags(layout: Layout, flags: DWORD) -> *mut u8 {
    if layout.align() <= MIN_ALIGN {
        return HeapAlloc(GetProcessHeap(), flags, layout.size()) as *mut u8;
    }

    let size = layout.size() + layout.align();
    let ptr = HeapAlloc(GetProcessHeap(), flags, size);
    if ptr.is_null() {
        ptr as *mut u8
    } else {
        align_ptr(ptr as *mut u8, layout.align())
    }
}

unsafe fn get_header<'a>(ptr: *mut u8) -> &'a mut Header {
    &mut *(ptr as *mut Header).offset(-1)
}

unsafe fn align_ptr(ptr: *mut u8, align: usize) -> *mut u8 {
    let aligned = ptr.add(align - (ptr as usize & (align - 1)));
    *get_header(aligned) = Header(ptr);
    aligned
}
