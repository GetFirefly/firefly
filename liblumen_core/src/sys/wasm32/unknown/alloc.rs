use core::alloc::prelude::*;
use core::ptr::NonNull;

use crate::locks::SpinLock;

static SYS_ALLOC_LOCK: SpinLock<dlmalloc::Dlmalloc> = SpinLock::new(dlmalloc::DLMALLOC_INIT);

#[inline]
pub fn alloc(layout: Layout) -> Result<MemoryBlock, AllocErr> {
    let mut allocator = SYS_ALLOC_LOCK.lock();
    let layout_size = layout.size();
    let ptr = (*allocator).malloc(layout_size, layout.align());
    NonNull::new(ptr).ok_or(AllocErr).map(|ptr| MemoryBlock {
        ptr,
        size: layout_size,
    })
}

#[inline]
pub fn alloc_zeroed(layout: Layout) -> Result<MemoryBlock, AllocErr> {
    let mut allocator = SYS_ALLOC_LOCK.lock();
    let layout_size = layout.size();
    let ptr = (*allocator).calloc(layout_size, layout.align());
    NonNull::new(ptr).ok_or(AllocErr).map(|ptr| MemoryBlock {
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
) -> Result<MemoryBlock, AllocErr> {
    if placement != ReallocPlacement::MayMove {
        return Err(AllocErr);
    }

    let mut allocator = SYS_ALLOC_LOCK.lock();
    let layout_size = layout.size();
    let new_ptr = (*allocator).realloc(ptr, layout_size, layout.align(), new_size);
    NonNull::new(new_ptr)
        .ok_or(AllocErr)
        .map(|ptr| MemoryBlock {
            ptr,
            size: layout_size,
        })
}

#[inline]
pub unsafe fn free(ptr: *mut u8, layout: Layout) {
    let mut allocator = SYS_ALLOC_LOCK.lock();
    (*allocator).free(ptr, layout.size(), layout.align());
}
