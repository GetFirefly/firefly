use core::ptr;
use core::alloc::AllocErr;

use winapi::um::memoryapi::{VirtualAlloc, VirtualFree};
use winapi::um::winnt::{MEM_COMMIT, MEM_RESERVE, MEM_RELEASE, MEM_RESET, PAGE_READWRITE};

use crate::mmap::MemMapResult;

/// Requests a new memory mapping from the OS.
#[inline]
pub unsafe fn map(_hint_ptr: *const usize, size: usize) -> MemMapResult<NonNull<u8>> {
    let ptr = VirtualAlloc(ptr::null_mut(), size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    NonNull::new(ptr as *mut u8).ok_or(AllocErr)
}

/// Like `map`, but hints to the kernel that the entire mapping should be reserved when first accessed.
///
/// NOTE: On Windows, the semantics of `map` and `map_reserved` are identical
#[inline(always)]
pub unsafe fn map_reserved(hint_ptr: *mut u8, size: usize) -> MemMapResult<NonNull<u8>> {
    // On Windows, memory mapping without committing, or reserving, the mapping is
    // not a meaningful distinction, a mapping must be committed to be used, and committing
    // it is the same as reserving it, so this function is identical in behavior to `map`
    mmap(hint_ptr, size)
}

/// Releases a memory region back to the OS
#[inline]
pub unsafe fn unmap(ptr: *mut u8, _size: usize) {
    VirtualFree(ptr, 0, MEM_RELEASE)
}

/// Marks the given memory region as unused without freeing it, letting the OS
/// reclaim its physical memory with the promise that we'll get it back (without
/// its contents) the next time it's accessed
#[inline]
pub unsafe fn discard(ptr: *mut u8, size: usize) {
    VirtualAlloc(ptr, size, MEM_RESET, PAGE_READWRITE);
}

/// Remap the memory mapping given by `ptr` and `old_size` to one with size `new_size`.
/// No guarantee is made that the new mapping will be remain in place
#[inline]
pub unsafe fn remap(ptr: *mut usize, old_size: usize, new_size: usize) -> MemMapResult<NonNull<u8>> {
    if new_size < old_size {
        return NonNull::new(ptr as *mut u8).ok_or(AllocErr);
    }

    // Acquire a new mapping
    let ret = map_reserved(ptr, new_size)?;
    // Copy over the old mapping to the new
    let dst_ptr = ret.as_ptr();
    ptr::copy_nonoverlapping(ptr, dst_ptr, old_size);
    // Free the old mapping
    unmap(ptr, old_size);

    return Ok(NonNull::new_unchecked(dst_ptr));
}
