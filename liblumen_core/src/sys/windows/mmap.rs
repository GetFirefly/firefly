use core::alloc::{AllocErr, Layout};
use core::mem;
use core::ptr::{self, NonNull};

use winapi::um::errhandlingapi::GetLastError;
use winapi::um::memoryapi::VmOfferPriorityVeryLow;
use winapi::um::memoryapi::{OfferVirtualMemory, VirtualAlloc, VirtualFree, VirtualQuery};
use winapi::um::winnt::{MEM_COMMIT, MEM_DECOMMIT, MEM_RELEASE, MEM_RESERVE};
use winapi::um::winnt::{PAGE_NOACCESS, PAGE_READWRITE};

use crate::alloc::alloc_utils;
use crate::sys::sysconf;

/// Requests a new memory mapping from the OS.
///
/// If `hint_ptr` is not a null pointer, it will be used to hint to the OS
/// where we would like the region mapped.
// While Windows makes a distinction between allocation granularity and page size (see
// https://msdn.microsoft.com/en-us/library/windows/desktop/ms724958(v=vs.85).aspx),
// VirtualAlloc only cares about allocation granularity for the pointer argument, not the size.
// Since we're passing null for the pointer, this doesn't affect us.
//
// NOTE: Windows can return many different error codes in different scenarios that all relate
// to being out of memory. Instead of trying to list them all, we assume that any error is an
// out-of-memory condition. This is fine so long as our code doesn't have a bug (that would,
// e.g., result in VirtualAlloc being called with invalid arguments). This isn't ideal, but
// during debugging, error codes can be printed here, so it's not the end of the world.
#[inline]
pub unsafe fn map(layout: Layout) -> Result<NonNull<u8>, AllocErr> {
    let page_size = sysconf::pagesize();
    let align = layout.align();

    if align <= page_size {
        // Alignment smaller than the system page size requires no
        // special work, as mmap aligns to the system page size by
        // default so all valid smaller alignments are automatically fulfilled
        let ptr = VirtualAlloc(
            ptr::null_mut(),
            layout.size(),
            MEM_COMMIT | MEM_RESERVE,
            PAGE_READWRITE,
        );
        return NonNull::new(ptr as *mut u8).ok_or(AllocErr);
    }

    // We have to handle alignment ourselves, so we reserve a larger region
    // and then only commit the aligned region we actually want. By setting
    // an appropriate protection level, we can ensure that the application fails
    // fast if access in the extra space occurs
    let base_size = layout.size();
    let size = alloc_utils::round_up_to_multiple_of(base_size, page_size);
    let extra = align - page_size;
    let padded = size + extra;

    let ptr = VirtualAlloc(ptr::null_mut(), padded, MEM_RESERVE, PAGE_NOACCESS);
    if ptr.is_null() {
        return Err(AllocErr);
    }

    // Calculate aligned address
    let addr = ptr as usize;
    let aligned_addr = alloc_utils::round_up_to_multiple_of(addr, align);
    let aligned_ptr = aligned_addr as *mut u8;

    // Commit the aligned region
    commit(aligned_ptr, base_size);

    Ok(NonNull::new_unchecked(aligned_ptr))
}

// Commits an already mapped region
//
// If the commit operation fails, this will cause a panic, so
// it is critical that you only call this with a pointer that was
// previously returned by VirtualAlloc
#[inline]
pub unsafe fn commit(ptr: *mut u8, size: usize) {
    let result = VirtualAlloc(ptr as *mut _, size, MEM_COMMIT, PAGE_READWRITE) as *mut u8;
    assert_ne!(
        result,
        ptr::null_mut(),
        "VirtualAlloc({:?}, {}) failed to commit reserved region (error: {})",
        ptr,
        size,
        result as usize
    );
}

// Decommits a mapped region
//
// This does not release the memory back to the OS, but
// rather makes the memory available for other uses until we
// commit the memory again
#[inline]
pub unsafe fn decommit(ptr: *mut u8, size: usize) {
    let result = VirtualFree(ptr as *mut _, size, MEM_DECOMMIT);
    assert_ne!(
        result,
        0,
        "decommit({:?}, {}) failed with {}",
        ptr,
        size,
        GetLastError()
    );
}

/// Releases a memory region back to the OS
#[inline]
pub unsafe fn unmap(ptr: *mut u8, layout: Layout) {
    use winapi::um::winnt::MEMORY_BASIC_INFORMATION;

    let page_size = sysconf::pagesize();
    let align = layout.align();

    if align <= page_size {
        // Nothing special to do here, just unmap
        VirtualFree(ptr as *mut _, 0, MEM_RELEASE);
        return;
    }

    // In order to call VirtualFree with the correct pointer
    // we have to look up the original pointer returned from
    // VirtualAlloc, which will not be `ptr`, as that was the
    // aligned pointer we returned originally. We use VirtualQuery
    // to look up the unaligned pointer
    let mut maybe_uninit_basic_information = mem::MaybeUninit::uninit();
    let basic_information_size = mem::size_of::<MEMORY_BASIC_INFORMATION>();
    let written_size = VirtualQuery(
        ptr as _,
        maybe_uninit_basic_information.as_mut_ptr(),
        basic_information_size,
    );
    assert_eq!(
        written_size,
        basic_information_size,
        "VirtualQuery({:?}) failed with {}",
        ptr,
        GetLastError()
    );
    let basic_information = maybe_uninit_basic_information.assume_init();
    VirtualFree(basic_information.AllocationBase, 0, MEM_RELEASE);
}

/// Remap the memory mapping given by `ptr` and `old_size` to one with size `new_size`.
/// No guarantee is made that the new mapping will be remain in place
///
/// NOTE: Windows can't unmap a portion of a previous mapping, so shrinking is not supported,
/// instead we currently just return the given pointer without making any changes, except
/// decommitting the extra memory to ensure the pages are not used and are not consuming physical
/// memory
#[inline]
pub unsafe fn remap(
    ptr: *mut u8,
    layout: Layout,
    new_size: usize,
) -> Result<NonNull<u8>, AllocErr> {
    let old_size = layout.size();
    let page_size = sysconf::pagesize();

    // If the new size is smaller and the extra space is at least one page,
    // offer up the extra pages to the operating system, if the extra space
    // less than a page, then we have nothing to do
    if new_size < old_size {
        let diff = old_size - new_size;
        if diff > page_size {
            let extra_ptr = ptr.offset((old_size - new_size) as isize);
            // Need to make sure the address we offer is page aligned
            let extra_addr = extra_ptr as usize;
            let aligned_addr = alloc_utils::round_up_to_multiple_of(extra_addr, page_size);
            if extra_addr != aligned_addr {
                // Need to adjust the diff
                let diff = diff - (aligned_addr - extra_addr);
                // If the diff is still at least one page, proceed
                if diff > page_size {
                    OfferVirtualMemory(aligned_addr as *mut _, diff, VmOfferPriorityVeryLow);
                }
            } else {
                OfferVirtualMemory(extra_ptr as *mut _, diff, VmOfferPriorityVeryLow);
            }
        }

        return Ok(NonNull::new_unchecked(ptr));
    }

    // Ensure that the new size is aligned with the old alignment
    let new_size = alloc_utils::round_up_to_multiple_of(new_size, layout.align());

    // Fallback to alloc/copy/dealloc
    remap_fallback(ptr, layout, new_size)
}

unsafe fn remap_fallback(
    ptr: *mut u8,
    layout: Layout,
    new_size: usize,
) -> Result<NonNull<u8>, AllocErr> {
    use core::cmp;

    // Create new mapping
    let new_ptr = map(layout)?;
    // Copy old data into new region
    let old_size = layout.size();
    ptr::copy_nonoverlapping(ptr, new_ptr.as_ptr(), cmp::min(old_size, new_size));
    // Destroy old mapping
    unmap(ptr, layout);

    Ok(new_ptr)
}
