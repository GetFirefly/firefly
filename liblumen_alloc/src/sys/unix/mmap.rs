use core::ptr::{self, NonNull};
use core::alloc::AllocErr;

mod constants {
    pub use libc::{PROT_READ, PROT_WRITE};

    pub use libc::MAP_FAILED;
    pub use libc::MAP_PRIVATE;

    #[cfg(target_os = "macos")]
    pub const MAP_ANONYMOUS: libc::c_int = libc::MAP_ANON;
    #[cfg(not(target_os = "macos"))]
    pub const MAP_ANONYMOUS: libc::c_int = libc::MAP_ANONYMOUS;

    pub use libc::MADV_FREE;
    pub use libc::MADV_WILLNEED;
}

use self::constants::*;

const MMAP_PROT: libc::c_int = PROT_READ | PROT_WRITE;
const MMAP_FLAGS: libc::c_int = MAP_PRIVATE | MAP_ANONYMOUS;

/// Requests a new memory mapping from the OS.
///
/// If `hint_ptr` is not a null pointer, it will be used to hint to the OS
/// where we would like the region mapped.
#[inline]
pub unsafe fn map(hint_ptr: *mut u8, size: usize) -> Result<NonNull<u8>, AllocErr> {
    let res = libc::mmap(hint_ptr as *mut libc::c_void, size, MMAP_PROT, MMAP_FLAGS, -1 as libc::c_int, 0);
    if res == MAP_FAILED {
        return Err(AllocErr);
    }

    NonNull::new(res as *mut u8).ok_or(AllocErr)
}

/// Like `map`, but hints to the kernel that the entire memory mapping should be reserved as soon
/// as it is accessed for the first time
#[allow(unused)]
#[inline]
pub unsafe fn map_reserved(hint_ptr: *mut u8, size: usize) -> Result<NonNull<u8>, AllocErr> {
    let res = libc::mmap(hint_ptr as *mut libc::c_void, size as libc::size_t, MMAP_PROT, MMAP_FLAGS, -1 as libc::c_int, 0);
    if res == MAP_FAILED {
        return Err(AllocErr);
    }

    // Indicate to the kernel that when first accessed, it should expect the entire mapping to be needed
    // This isn't quite the same as physical memory reservation, but as close as we can get in practice
    libc::madvise(res, size as libc::size_t, MADV_WILLNEED);

    NonNull::new(res as *mut u8).ok_or(AllocErr)
}

/// Releases a memory region back to the OS
#[inline]
pub unsafe fn unmap(ptr: *mut u8, size: usize) {
    libc::munmap(ptr as *mut libc::c_void, size as libc::size_t);
}

/// Marks the given memory region as unused without freeing it, letting the OS
/// reclaim its physical memory with the promise that we'll get it back (without
/// its contents) the next time it's accessed
#[allow(unused)]
#[inline]
pub unsafe fn discard(ptr: *mut u8, size: usize) {
    // If unsupported, we may have to add conditional compilation to use MADV_DONTNEED instead
    libc::madvise(ptr as *mut libc::c_void, size as libc::size_t, MADV_FREE);
}

/// Remap the memory mapping given by `ptr` and `old_size` to one with size `new_size`.
/// No guarantee is made that the new mapping will be remain in place
#[allow(unused)]
#[inline]
#[cfg(any(target_os = "linux", target_os = "emscripten", target_os = "android"))]
pub unsafe fn remap(ptr: *mut u8, old_size: usize, new_size: usize) -> Result<NonNull<u8>, AllocErr> {
    let new_seg = libc::mremap(ptr as *mut libc::c_void, old_size, new_size, libc::MREMAP_MAYMOVE);
    let new_seg = match new_seg {
        MAP_FAILED => ptr::null(),
        other => other
    };
    NonNull::new(new_seg as *mut _).ok_or(AllocErr)
}

#[allow(unused)]
#[inline]
#[cfg(any(target_os = "freebsd", target_os = "netbsd"))]
pub unsafe fn remap(ptr: *mut u8, old_size: usize, new_size: usize) -> Result<NonNull<u8>, AllocErr> {
    if new_size < old_size {
        return NonNull::new(ptr).ok_or(AllocErr);
    }

    let new_seg = libc::mremap(ptr as *mut libc::c_void, old_size, ptr::null_mut(), new_size, 0 as libc::c_int);
    let new_seg = match new_seg {
        MAP_FAILED => ptr::null(),
        other => other
    };
    NonNull::new(new_seg as *mut _).ok_or(AllocErr)
}

/// remap the memory mapping given by `ptr` and `old_size` to one with size `new_size`.
/// No guarantee is made that the new mapping will be remain in place
#[allow(unused)]
#[inline]
#[cfg(all(
    not(target_os = "linux"),
    not(target_os = "emscripten"),
    not(target_os = "android"),
    not(target_os = "freebsd"),
    not(target_os = "netbsd"),
))]
pub unsafe fn remap(ptr: *mut u8, old_size: usize, new_size: usize) -> Result<NonNull<u8>, AllocErr> {
    if new_size < old_size {
        return NonNull::new(ptr).ok_or(AllocErr);
    }

    // Try and map the extra space at the end of the old mapping
    let hint_ptr = ((ptr as usize) + old_size) as *mut libc::c_void;
    let ret = libc::mmap(hint_ptr, new_size - old_size, MMAP_PROT, MMAP_FLAGS, -1 as libc::c_int, 0);

    // Unable to map any new memory
    if ret == MAP_FAILED {
        return Err(AllocErr);
    }

    // We have the memory, but not where we wanted it
    if ret != hint_ptr {
        unmap(ret as *mut _, new_size - old_size);
        // Acquire a fresh mapping
        let ret = map(ptr::null_mut(), new_size)?;
        // Copy over the old mapping to the new
        let dst_ptr = ret.as_ptr();
        ptr::copy_nonoverlapping(ptr, dst_ptr, old_size);
        // Free the old mapping
        unmap(ptr, old_size);

        return Ok(NonNull::new_unchecked(dst_ptr));
    }

    // We were able to remap the original mapping
    NonNull::new(ret as *mut _).ok_or(AllocErr)
}
