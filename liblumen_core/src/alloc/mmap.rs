///! This module provides a cross-platform mmap interface.
///!
///! On platforms without actual memory mapping primitives,
///! this delegates to the system allocator
use core::ptr::NonNull;

use crate::alloc::prelude::*;

#[cfg(not(has_mmap))]
use crate::sys::alloc as sys_alloc;
#[cfg(has_mmap)]
use crate::sys::mmap;

/// Creates a memory mapping for the given `Layout`
#[cfg(has_mmap)]
#[inline]
pub unsafe fn map(layout: Layout) -> Result<NonNull<u8>, AllocError> {
    mmap::map(layout).map(|(ptr, _)| ptr)
}

/// Creates a memory mapping for the given `Layout`
#[cfg(not(has_mmap))]
#[inline]
pub unsafe fn map(layout: Layout) -> Result<NonNull<u8>, AllocError> {
    sys_alloc::allocate(layout).map(|ptr| ptr.cast())
}

/// Creates a memory mapping specifically set up to behave like a stack
#[cfg(has_mmap)]
#[inline]
pub unsafe fn map_stack(pages: usize) -> Result<NonNull<u8>, AllocError> {
    mmap::map_stack(pages)
}

/// Creates a memory mapping specifically set up to behave like a stack
///
/// NOTE: This is a fallback implementation, so no guard page is present,
/// and it is implemented the same as `map`
#[cfg(not(has_mmap))]
#[inline]
pub unsafe fn map_stack(pages: usize) -> Result<NonNull<u8>, AllocError> {
    let page_size = crate::sys::sysconf::pagesize();
    let (layout, _offset) = Layout::from_size_align(page_size, page_size)
        .unwrap()
        .repeat(pages)
        .unwrap();

    sys_alloc::allocate(layout).map(|ptr| ptr.cast())
}

/// Remaps a mapping given a pointer to the mapping, the layout which created it, and the new size
#[cfg(has_mmap)]
#[inline]
pub unsafe fn remap(
    ptr: *mut u8,
    layout: Layout,
    new_size: usize,
) -> Result<NonNull<u8>, AllocError> {
    mmap::remap(ptr, layout, new_size).map(|(ptr, _)| ptr)
}

/// Remaps a mapping given a pointer to the mapping, the layout which created it, and the new size
#[cfg(not(has_mmap))]
#[inline]
pub unsafe fn remap(
    ptr: *mut u8,
    layout: Layout,
    new_size: usize,
) -> Result<NonNull<u8>, AllocError> {
    let old_layout = layout;
    let new_layout =
        Layout::from_size_align(new_size, old_layout.align()).map_err(|_| AllocError)?;
    let non_null = NonNull::new(ptr).ok_or(AllocError)?;

    if layout.size() < new_size {
        sys_alloc::grow(non_null, old_layout, new_layout)
    } else {
        sys_alloc::shrink(non_null, old_layout, new_layout)
    }
    .map(|ptr| ptr.cast())
}

/// Destroys a mapping given a pointer to the mapping and the layout which created it
#[cfg(has_mmap)]
#[inline]
pub unsafe fn unmap(ptr: *mut u8, layout: Layout) {
    mmap::unmap(ptr, layout);
}

/// Destroys a mapping given a pointer to the mapping and the layout which created it
#[cfg(not(has_mmap))]
#[inline]
pub unsafe fn unmap(ptr: *mut u8, layout: Layout) {
    sys_alloc::deallocate(NonNull::new(ptr).unwrap(), layout);
}
