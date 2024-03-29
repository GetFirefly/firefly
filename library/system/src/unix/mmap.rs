use core::alloc::{AllocError, Layout};
#[cfg(all(
    not(target_os = "linux"),
    not(target_os = "emscripten"),
    not(target_os = "android"),
    not(target_os = "freebsd"),
    not(target_os = "netbsd"),
))]
use core::cmp;
use core::intrinsics::unlikely;
use core::ptr::{self, NonNull};

use super::sysconf;
use crate::alloc::round_up_to_multiple_of;

mod constants {
    pub use libc::{PROT_NONE, PROT_READ, PROT_WRITE};

    #[cfg(not(any(
        target_os = "freebsd",
        target_os = "dragonfly",
        target_vendor = "apple"
    )))]
    pub use libc::MAP_STACK;

    pub use libc::MAP_FAILED;
    pub use libc::MAP_PRIVATE;

    #[cfg(target_os = "macos")]
    pub const MAP_ANONYMOUS: libc::c_int = libc::MAP_ANON;
    #[cfg(not(target_os = "macos"))]
    pub const MAP_ANONYMOUS: libc::c_int = libc::MAP_ANONYMOUS;

    pub use libc::MADV_FREE;
    pub use libc::MADV_WILLNEED;

    #[cfg(any(target_os = "freebsd", target_os = "netbsd"))]
    const MAP_ALIGNMENT_SHIFT: libc::c_int = 24;
}

use self::constants::*;

const MMAP_PROT: libc::c_int = PROT_READ | PROT_WRITE;
const MMAP_FLAGS: libc::c_int = MAP_PRIVATE | MAP_ANONYMOUS;

const GUARD_PROT: libc::c_int = PROT_NONE;
const STACK_PROT: libc::c_int = PROT_READ | PROT_WRITE;

#[cfg(not(any(
    target_os = "freebsd",
    target_os = "dragonfly",
    target_vendor = "apple"
)))]
const STACK_FLAGS: libc::c_int = MAP_STACK | MMAP_FLAGS;
#[cfg(any(
    target_os = "freebsd",
    target_os = "dragonfly",
    target_vendor = "apple"
))]
const STACK_FLAGS: libc::c_int = MMAP_FLAGS;

/// Requests a new memory mapping from the OS.
///
/// If `hint_ptr` is not a null pointer, it will be used to hint to the OS
/// where we would like the region mapped.
#[inline]
pub unsafe fn map(layout: Layout) -> Result<(NonNull<u8>, usize), AllocError> {
    let page_size = sysconf::page_size();
    let size = round_up_to_multiple_of(layout.size(), page_size);
    let align = layout.align();

    if align <= page_size {
        // Alignment smaller than the system page size requires no
        // special work, as mmap aligns to the system page size by
        // default so all valid smaller alignments are automatically fulfilled
        return map_internal(ptr::null_mut(), size);
    }

    let extra = align - page_size;

    // To avoid wasting space, we unmap the unused portions of
    // this initial memory mapping, and return the aligned region
    let ptr = map_internal(ptr::null_mut(), size + extra).map(|(nn, _)| nn.as_ptr())?;
    let addr = ptr as usize;
    let aligned_addr = round_up_to_multiple_of(addr, align);
    let aligned_ptr = aligned_addr as *mut u8;

    // Unmap the unused prefix region
    let prefix_size = aligned_addr - addr;
    if prefix_size > 0 {
        unmap_internal(addr as *mut u8, prefix_size);
    }
    // Unmap the unused suffix region
    let suffix_size = extra - prefix_size;
    if suffix_size > 0 {
        unmap_internal((aligned_addr + size) as *mut u8, suffix_size);
    }

    // Commit the precise memory region for the requested layout
    let commit_size = layout.size();
    commit(aligned_ptr, commit_size);

    Ok((NonNull::new_unchecked(aligned_ptr), commit_size))
}

#[inline]
pub unsafe fn map_stack(pages: usize) -> Result<NonNull<u8>, AllocError> {
    // Stacks must be at least 1 page + 1 guard page
    let page_size = sysconf::page_size();
    let stack_size = page_size * (pages + 1);
    let size = round_up_to_multiple_of(stack_size, page_size);

    let res = libc::mmap(
        ptr::null_mut(),
        size,
        STACK_PROT,
        STACK_FLAGS,
        -1 as libc::c_int,
        0,
    );

    if res == MAP_FAILED {
        return Err(AllocError);
    }

    // Set up guard page
    if 0 == libc::mprotect(res as *mut libc::c_void, page_size, GUARD_PROT) {
        Ok(NonNull::new_unchecked(res as *mut u8))
    } else {
        Err(AllocError)
    }
}

#[inline(always)]
unsafe fn map_internal(hint_ptr: *mut u8, size: usize) -> Result<(NonNull<u8>, usize), AllocError> {
    let res = libc::mmap(
        hint_ptr as *mut libc::c_void,
        size,
        MMAP_PROT,
        MMAP_FLAGS,
        -1 as libc::c_int,
        0,
    );
    if res == MAP_FAILED {
        return Err(AllocError);
    }

    Ok((NonNull::new_unchecked(res as *mut u8), size))
}

#[inline(always)]
unsafe fn commit(ptr: *mut u8, size: usize) {
    libc::madvise(ptr as *mut _, size, MADV_WILLNEED);
}

#[allow(unused)]
#[inline(always)]
unsafe fn decommit(ptr: *mut u8, size: usize) {
    // If unsupported, we may have to add conditional compilation to use MADV_DONTNEED instead
    libc::madvise(ptr as *mut _, size, MADV_FREE);
}

/// Releases a memory region back to the OS
#[inline(always)]
pub unsafe fn unmap(ptr: *mut u8, layout: Layout) {
    unmap_internal(ptr, layout.size());
}

#[inline(always)]
unsafe fn unmap_internal(ptr: *mut u8, size: usize) {
    libc::munmap(ptr as *mut _, size as libc::size_t);
}

/// Remaps the memory mapping at `ptr` using the alignment of `layout` and `new_size`
///
/// NOTE: No guarantee is made that the new mapping will be remain in place
#[inline]
pub unsafe fn remap(
    ptr: *mut u8,
    layout: Layout,
    new_size: usize,
) -> Result<(NonNull<u8>, usize), AllocError> {
    let old_size = layout.size();
    let align = layout.align();
    let new_size = round_up_to_multiple_of(new_size, align);

    if unlikely(new_size < old_size) {
        return Ok((NonNull::new_unchecked(ptr), new_size));
    }

    remap_internal(ptr, old_size, align, new_size)
}

#[inline]
#[cfg(any(target_os = "linux", target_os = "emscripten", target_os = "android"))]
unsafe fn remap_internal(
    ptr: *mut u8,
    old_size: usize,
    _align: usize,
    new_size: usize,
) -> Result<(NonNull<u8>, usize), AllocError> {
    let new_seg = libc::mremap(ptr as *mut _, old_size, new_size, libc::MREMAP_MAYMOVE);
    if new_seg == MAP_FAILED {
        return Err(AllocError);
    }
    Ok((NonNull::new_unchecked(new_seg as *mut _), new_size))
}

/// Remaps the memory mapping at `ptr` using the alignment of `layout` and `new_size`
///
/// NOTE: No guarantee is made that the new mapping will be remain in place
#[inline]
#[cfg(any(target_os = "freebsd", target_os = "netbsd"))]
unsafe fn remap_internal(
    ptr: *mut u8,
    old_size: usize,
    _align: usize,
    new_size: usize,
) -> Result<(NonNull<u8>, usize), AllocError> {
    let new_seg = libc::mremap(ptr as *mut _, old_size, 0, new_size, 0 as libc::c_int);
    if new_seg == MAP_FAILED {
        return Err(AllocError);
    }
    Ok((NonNull::new_unchecked(new_seg as *mut _), new_size))
}

/// Remaps the memory mapping at `ptr` using the alignment of `layout` and `new_size`
///
/// NOTE: No guarantee is made that the new mapping will be remain in place
#[inline]
#[cfg(all(
    not(target_os = "linux"),
    not(target_os = "emscripten"),
    not(target_os = "android"),
    not(target_os = "freebsd"),
    not(target_os = "netbsd"),
))]
unsafe fn remap_internal(
    ptr: *mut u8,
    old_size: usize,
    align: usize,
    new_size: usize,
) -> Result<(NonNull<u8>, usize), AllocError> {
    // Try and map the extra space at the end of the old mapping
    let hint_ptr = ((ptr as usize) + old_size) as *mut libc::c_void;
    let extend_size = new_size - old_size;
    let ret = libc::mmap(
        hint_ptr,
        extend_size,
        MMAP_PROT,
        MMAP_FLAGS,
        -1 as libc::c_int,
        0,
    );

    // Unable to map any new memory
    if ret == MAP_FAILED {
        return Err(AllocError);
    }

    // We have the memory, but not where we wanted it
    if ret != hint_ptr {
        // Unmap the mapping we just made
        unmap_internal(ret as *mut _, extend_size);
        // Fallback to alloc+copy+free
        let layout = Layout::from_size_align_unchecked(old_size, align);
        return remap_fallback(ptr, layout, new_size);
    }

    // We were able to remap the original mapping
    Ok((NonNull::new_unchecked(ret as *mut _), new_size))
}

#[inline]
#[cfg(all(
    not(target_os = "linux"),
    not(target_os = "emscripten"),
    not(target_os = "android"),
    not(target_os = "freebsd"),
    not(target_os = "netbsd"),
))]
unsafe fn remap_fallback(
    ptr: *mut u8,
    layout: Layout,
    new_size: usize,
) -> Result<(NonNull<u8>, usize), AllocError> {
    // Allocate new mapping
    let new_layout = Layout::from_size_align(new_size, layout.align()).expect("invalid layout");
    let (new_ptr, new_ptr_size) = map(new_layout)?;
    // Copy over the old mapping to the new
    let old_size = layout.size();
    ptr::copy_nonoverlapping(ptr, new_ptr.as_ptr(), cmp::min(old_size, new_size));
    // Free the old mapping
    unmap(ptr, layout);

    Ok((new_ptr, new_ptr_size))
}
