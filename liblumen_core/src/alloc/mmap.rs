///! This module provides a cross-platform mmap interface.
///!
///! On platforms without actual memory mapping primitives,
///! this delegates to the system allocator
use core::alloc::{AllocErr, Layout};
use core::ptr::NonNull;

#[cfg(not(has_mmap))]
use crate::sys::alloc as sys_alloc;
#[cfg(has_mmap)]
use crate::sys::mmap;

/// Creates a memory mapping for the given `Layout`
#[cfg(has_mmap)]
#[inline]
pub unsafe fn map(layout: Layout) -> Result<NonNull<u8>, AllocErr> {
    mmap::map(layout)
}

/// Creates a memory mapping for the given `Layout`
#[cfg(not(has_mmap))]
#[inline]
pub unsafe fn map(layout: Layout) -> Result<NonNull<u8>, AllocErr> {
    sys_alloc::alloc(layout)
}

/// Remaps a mapping given a pointer to the mapping, the layout which created it, and the new size
#[cfg(has_mmap)]
#[inline]
pub unsafe fn remap(
    ptr: *mut u8,
    layout: Layout,
    new_size: usize,
) -> Result<NonNull<u8>, AllocErr> {
    mmap::remap(ptr, layout, new_size)
}

/// Remaps a mapping given a pointer to the mapping, the layout which created it, and the new size
#[cfg(not(has_mmap))]
#[inline]
pub unsafe fn remap(
    ptr: *mut u8,
    layout: Layout,
    new_size: usize,
) -> Result<NonNull<u8>, AllocErr> {
    sys_alloc::realloc(ptr, layout, new_size)
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
    sys_alloc::free(ptr, layout);
}
