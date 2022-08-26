use alloc::alloc::{AllocError, Allocator, GlobalAlloc, Layout};
use core::ptr::{self, NonNull};

use firefly_system::arch::alloc as sys_alloc;

/// This allocator acts as the system allocator.
///
/// Depending on the target, that may be the actual system allocator, or we
/// may provide own implementation.
#[derive(Debug, Copy, Clone)]
pub struct System;
unsafe impl Sync for System {}
unsafe impl Send for System {}

unsafe impl Allocator for System {
    #[inline]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        sys_alloc::allocate(layout)
    }

    #[inline]
    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        sys_alloc::allocate_zeroed(layout)
    }

    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        sys_alloc::deallocate(ptr, layout);
    }

    #[inline]
    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        sys_alloc::grow(ptr, old_layout, new_layout)
    }

    #[inline]
    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        sys_alloc::grow_zeroed(ptr, old_layout, new_layout)
    }

    #[inline]
    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        sys_alloc::shrink(ptr, old_layout, new_layout)
    }
}

unsafe impl GlobalAlloc for System {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        sys_alloc::allocate(layout)
            .map(|ptr| ptr.as_mut_ptr())
            .unwrap_or(ptr::null_mut())
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        sys_alloc::allocate_zeroed(layout)
            .map(|ptr| ptr.as_mut_ptr())
            .unwrap_or(ptr::null_mut())
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        sys_alloc::deallocate(NonNull::new(ptr).unwrap(), layout);
    }

    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, old_layout: Layout, new_size: usize) -> *mut u8 {
        NonNull::new(ptr)
            .and_then(|non_null| {
                Layout::from_size_align(new_size, old_layout.align())
                    .ok()
                    .and_then(|new_layout| {
                        if old_layout.size() <= new_size {
                            sys_alloc::grow(non_null, old_layout, new_layout)
                        } else {
                            sys_alloc::shrink(non_null, old_layout, new_layout)
                        }
                        .ok()
                        .map(|ptr| ptr.as_mut_ptr())
                    })
            })
            .unwrap_or(ptr::null_mut())
    }
}
