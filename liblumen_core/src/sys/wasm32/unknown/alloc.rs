use core::alloc::{AllocErr, Layout};
use core::ptr::NonNull;

use crate::locks::SpinLock;

static SYS_ALLOC_LOCK: SpinLock<dlmalloc::Dlmalloc> = SpinLock::new(dlmalloc::DLMALLOC_INIT);

#[inline]
pub unsafe fn alloc(layout: Layout) -> Result<(NonNull<u8>, usize), AllocErr> {
    let mut allocator = SYS_ALLOC_LOCK.lock();
    let layout_size = layout.size();
    let ptr = (*allocator).malloc(layout_size, layout.align());
    NonNull::new(ptr)
        .ok_or(AllocErr)
        .map(|ptr| (ptr, layout_size))
}

#[inline]
pub unsafe fn alloc_zeroed(layout: Layout) -> Result<(NonNull<u8>, usize), AllocErr> {
    let mut allocator = SYS_ALLOC_LOCK.lock();
    let layout_size = layout.size();
    let ptr = (*allocator).calloc(layout_size, layout.align());
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
    let mut allocator = SYS_ALLOC_LOCK.lock();
    let layout_size = layout.size();
    let new_ptr = (*allocator).realloc(ptr, layout_size, layout.align(), new_size);
    NonNull::new(new_ptr)
        .ok_or(AllocErr)
        .map(|ptr| (ptr, layout_size))
}

#[inline]
pub unsafe fn free(ptr: *mut u8, layout: Layout) {
    let mut allocator = SYS_ALLOC_LOCK.lock();
    (*allocator).free(ptr, layout.size(), layout.align());
}
