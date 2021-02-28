use core::alloc::{AllocError, Layout, MemoryBlock};

use core::ptr::NonNull;

use crate::locks::SpinLock;

static SYS_ALLOC_LOCK: SpinLock<dlmalloc::Dlmalloc> = SpinLock::new(dlmalloc::DLMALLOC_INIT);

#[inline]
pub fn alloc(layout: Layout) -> Result<MemoryBlock, AllocError> {
    let mut allocator = SYS_ALLOC_LOCK.lock();
    let layout_size = layout.size();
    let ptr = unsafe { (*allocator).malloc(layout_size, layout.align()) };
    NonNull::new(ptr).ok_or(AllocError).map(|ptr| MemoryBlock {
        ptr,
        size: layout_size,
    })
}

#[inline]
pub fn alloc_zeroed(layout: Layout) -> Result<MemoryBlock, AllocError> {
    let mut allocator = SYS_ALLOC_LOCK.lock();
    let layout_size = layout.size();
    let ptr = unsafe { (*allocator).calloc(layout_size, layout.align()) };
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
    self::realloc(ptr, layout, new_size, placement)
}

#[inline]
pub unsafe fn realloc(
    ptr: *mut u8,
    layout: Layout,
    new_size: usize,
) -> Result<MemoryBlock, AllocError> {
    let mut allocator = SYS_ALLOC_LOCK.lock();
    let layout_size = layout.size();
    let new_ptr = (*allocator).realloc(ptr, layout_size, layout.align(), new_size);
    NonNull::new(new_ptr)
        .ok_or(AllocError)
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
