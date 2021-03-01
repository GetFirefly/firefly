use core::alloc::{AllocError, Layout};
use core::ptr::{self, NonNull};

use dlmalloc::Dlmalloc;

use crate::locks::SpinLock;

static SYS_ALLOC_LOCK: SpinLock<Dlmalloc> = SpinLock::new(Dlmalloc::new());

#[inline]
pub fn allocate(layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
    let mut allocator = SYS_ALLOC_LOCK.lock();
    let layout_size = layout.size();
    let ptr = unsafe { (*allocator).malloc(layout_size, layout.align()) };
    NonNull::new(ptr)
        .ok_or(AllocError)
        .map(|ptr| NonNull::slice_from_raw_parts(ptr, layout_size))
}

#[inline]
pub fn allocate_zeroed(layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
    let mut allocator = SYS_ALLOC_LOCK.lock();
    let layout_size = layout.size();
    let ptr = unsafe { (*allocator).calloc(layout_size, layout.align()) };
    NonNull::new(ptr)
        .ok_or(AllocError)
        .map(|ptr| NonNull::slice_from_raw_parts(ptr, layout_size))
}

#[inline]
pub unsafe fn deallocate(ptr: NonNull<u8>, layout: Layout) {
    let mut allocator = SYS_ALLOC_LOCK.lock();
    (*allocator).free(ptr.as_ptr(), layout.size(), layout.align());
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

#[inline]
unsafe fn reallocate(
    ptr: NonNull<u8>,
    old_layout: Layout,
    new_layout: Layout,
) -> Result<NonNull<[u8]>, AllocError> {
    // TODO handle changed align
    assert!(old_layout.align() == new_layout.align());
    let mut allocator = SYS_ALLOC_LOCK.lock();
    let old_size = old_layout.size();
    let old_align = old_layout.align();
    let new_size = new_layout.size();
    let new_ptr = (*allocator).realloc(ptr.as_ptr(), old_size, old_align, new_size);
    NonNull::new(new_ptr)
        .ok_or(AllocError)
        .map(|ptr| NonNull::slice_from_raw_parts(ptr, new_size))
}
