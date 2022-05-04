use core::cmp;
use core::ptr::{self, NonNull};

use crate::alloc::prelude::*;
use crate::sys::alloc as sys_alloc;

/// This allocator acts as the system allocator, depending
/// on the target, that may be the actual system allocator,
/// or our own implementation.
#[derive(Debug, Copy, Clone)]
pub struct SysAlloc;
unsafe impl Sync for SysAlloc {}
unsafe impl Send for SysAlloc {}

static mut SINGLETON: SysAlloc = SysAlloc;

impl SysAlloc {
    #[inline(always)]
    pub fn get() -> &'static SysAlloc {
        unsafe { &SINGLETON }
    }

    #[inline(always)]
    pub fn get_mut() -> &'static mut SysAlloc {
        unsafe { &mut SINGLETON }
    }
}

unsafe impl Allocator for SysAlloc {
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

unsafe impl GlobalAlloc for SysAlloc {
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

/// Fallback for realloc that allocates a new region, copies old data
/// into the new region, and frees the old region.
#[inline]
pub unsafe fn realloc_fallback(
    old_ptr: NonNull<u8>,
    old_layout: Layout,
    new_layout: Layout,
) -> Result<NonNull<[u8]>, AllocError> {
    use core::intrinsics::unlikely;

    let old_size = old_layout.size();
    let new_size = new_layout.size();

    if unlikely(old_size == new_size) {
        return Ok(NonNull::slice_from_raw_parts(old_ptr, new_size));
    }

    let align = old_layout.align();
    let new_layout = Layout::from_size_align(new_size, align).expect("invalid layout");

    // Allocate new region, using mmap for allocations larger than page size
    let new_ptr = sys_alloc::allocate(new_layout)?;
    // Copy old region to new region
    ptr::copy_nonoverlapping(
        old_ptr.as_ptr(),
        new_ptr.as_mut_ptr(),
        cmp::min(old_size, new_ptr.len()),
    );
    // Free old region
    sys_alloc::deallocate(old_ptr, old_layout);

    Ok(new_ptr)
}
