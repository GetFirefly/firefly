use core::cmp;
use core::ptr::{self, NonNull};

use crate::alloc::alloc_handle::{AsAllocHandle, Global};
use crate::alloc::{AllocErr, AllocRef, GlobalAlloc, Layout};
use crate::sys::alloc as sys_alloc;

use super::StaticAlloc;

/// This allocator acts as the system allocator, depending
/// on the target, that may be the actual system allocator,
/// or our own implementation.
#[derive(Debug, Copy, Clone)]
pub struct SysAlloc;

unsafe impl AllocRef for SysAlloc {
    #[inline]
    unsafe fn alloc(&mut self, layout: Layout) -> Result<(NonNull<u8>, usize), AllocErr> {
        sys_alloc::alloc(layout)
    }

    #[inline]
    unsafe fn alloc_zeroed(&mut self, layout: Layout) -> Result<(NonNull<u8>, usize), AllocErr> {
        sys_alloc::alloc_zeroed(layout)
    }

    #[inline]
    unsafe fn dealloc(&mut self, ptr: NonNull<u8>, layout: Layout) {
        sys_alloc::free(ptr.as_ptr(), layout);
    }

    #[inline]
    unsafe fn realloc(
        &mut self,
        ptr: NonNull<u8>,
        layout: Layout,
        new_size: usize,
    ) -> Result<(NonNull<u8>, usize), AllocErr> {
        sys_alloc::realloc(ptr.as_ptr(), layout, new_size)
    }
}

unsafe impl GlobalAlloc for SysAlloc {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        sys_alloc::alloc(layout)
            .map(|(nn, _)| nn.as_ptr())
            .unwrap_or(ptr::null_mut())
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        sys_alloc::alloc_zeroed(layout)
            .map(|(nn, _)| nn.as_ptr())
            .unwrap_or(ptr::null_mut())
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        sys_alloc::free(ptr, layout);
    }

    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        sys_alloc::realloc(ptr, layout, new_size)
            .map(|(nn, _)| nn.as_ptr())
            .unwrap_or(ptr::null_mut())
    }
}

// Used by the StaticAlloc impl
static mut SYS_ALLOC: SysAlloc = SysAlloc;

unsafe impl StaticAlloc for SysAlloc {
    #[inline]
    unsafe fn static_ref() -> &'static Self {
        &SYS_ALLOC
    }
    #[inline]
    unsafe fn static_mut() -> &'static mut Self {
        &mut SYS_ALLOC
    }
}

impl AsAllocHandle<'static> for SysAlloc {
    type Handle = Global<Self>;

    #[inline]
    fn as_alloc_handle(&'static self) -> Self::Handle {
        Global::new()
    }
}

/// Fallback for realloc that allocates a new region, copies old data
/// into the new region, and frees the old region.
#[inline]
pub unsafe fn realloc_fallback(
    ptr: *mut u8,
    old_layout: Layout,
    new_size: usize,
) -> Result<(NonNull<u8>, usize), AllocErr> {
    use core::intrinsics::unlikely;

    let old_size = old_layout.size();

    if unlikely(old_size == new_size) {
        return Ok((NonNull::new_unchecked(ptr), new_size));
    }

    let align = old_layout.align();
    let new_layout = Layout::from_size_align(new_size, align).expect("invalid layout");

    // Allocate new region, using mmap for allocations larger than page size
    let (new_ptr, new_ptr_size) = sys_alloc::alloc(new_layout)?;
    // Copy old region to new region
    ptr::copy_nonoverlapping(ptr, new_ptr.as_ptr(), cmp::min(old_size, new_size));
    // Free old region
    sys_alloc::free(ptr, old_layout);

    Ok((new_ptr, new_ptr_size))
}
