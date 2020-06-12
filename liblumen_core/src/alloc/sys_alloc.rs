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

unsafe impl AllocRef for SysAlloc {
    #[inline]
    fn alloc(&mut self, layout: Layout, init: AllocInit) -> Result<MemoryBlock, AllocErr> {
        match init {
            AllocInit::Uninitialized => sys_alloc::alloc(layout),
            AllocInit::Zeroed => sys_alloc::alloc_zeroed(layout),
        }
    }

    #[inline]
    unsafe fn dealloc(&mut self, ptr: NonNull<u8>, layout: Layout) {
        sys_alloc::free(ptr.as_ptr(), layout);
    }

    #[inline]
    unsafe fn grow(
        &mut self,
        ptr: NonNull<u8>,
        layout: Layout,
        new_size: usize,
        placement: ReallocPlacement,
        init: AllocInit,
    ) -> Result<MemoryBlock, AllocErr> {
        sys_alloc::grow(ptr.as_ptr(), layout, new_size, placement, init)
    }

    #[inline]
    unsafe fn shrink(
        &mut self,
        ptr: NonNull<u8>,
        layout: Layout,
        new_size: usize,
        placement: ReallocPlacement,
    ) -> Result<MemoryBlock, AllocErr> {
        sys_alloc::shrink(ptr.as_ptr(), layout, new_size, placement)
    }
}

unsafe impl GlobalAlloc for SysAlloc {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        sys_alloc::alloc(layout)
            .map(|block| block.ptr.as_ptr())
            .unwrap_or(ptr::null_mut())
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        sys_alloc::alloc_zeroed(layout)
            .map(|block| block.ptr.as_ptr())
            .unwrap_or(ptr::null_mut())
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        sys_alloc::free(ptr, layout);
    }

    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if layout.size() <= new_size {
            sys_alloc::grow(
                ptr,
                layout,
                new_size,
                ReallocPlacement::MayMove,
                AllocInit::Uninitialized,
            )
            .map(|block| block.ptr.as_ptr())
            .unwrap_or(ptr::null_mut())
        } else {
            sys_alloc::shrink(ptr, layout, new_size, ReallocPlacement::MayMove)
                .map(|block| block.ptr.as_ptr())
                .unwrap_or(ptr::null_mut())
        }
    }
}

/// Fallback for realloc that allocates a new region, copies old data
/// into the new region, and frees the old region.
#[inline]
pub unsafe fn realloc_fallback(
    ptr: *mut u8,
    old_layout: Layout,
    new_size: usize,
) -> Result<MemoryBlock, AllocErr> {
    use core::intrinsics::unlikely;

    let old_size = old_layout.size();

    if unlikely(old_size == new_size) {
        return Ok(MemoryBlock {
            ptr: NonNull::new_unchecked(ptr),
            size: new_size,
        });
    }

    let align = old_layout.align();
    let new_layout = Layout::from_size_align(new_size, align).expect("invalid layout");

    // Allocate new region, using mmap for allocations larger than page size
    let block = sys_alloc::alloc(new_layout)?;
    // Copy old region to new region
    ptr::copy_nonoverlapping(ptr, block.ptr.as_ptr(), cmp::min(old_size, block.size));
    // Free old region
    sys_alloc::free(ptr, old_layout);

    Ok(block)
}
