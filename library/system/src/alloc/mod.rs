use core::alloc::{AllocError, Layout};
use core::ptr::{self, NonNull};

/// Fallback for realloc that allocates a new region, copies old data
/// into the new region, and frees the old region.
#[inline]
pub unsafe fn realloc_fallback(
    old_ptr: NonNull<u8>,
    old_layout: Layout,
    new_layout: Layout,
) -> Result<NonNull<[u8]>, AllocError> {
    use core::cmp::min;
    use core::intrinsics::unlikely;

    let old_size = old_layout.size();
    let new_size = new_layout.size();

    if unlikely(old_size == new_size) {
        return Ok(NonNull::slice_from_raw_parts(old_ptr, new_size));
    }

    let align = old_layout.align();
    let new_layout = Layout::from_size_align(new_size, align).expect("invalid layout");

    // Allocate new region, using mmap for allocations larger than page size
    let new_ptr = crate::arch::alloc::allocate(new_layout)?;
    // Copy old region to new region
    ptr::copy_nonoverlapping(
        old_ptr.as_ptr(),
        new_ptr.as_mut_ptr(),
        min(old_size, new_ptr.len()),
    );
    // Free old region
    crate::arch::alloc::deallocate(old_ptr, old_layout);

    Ok(new_ptr)
}

/// Returns `size` rounded up to a multiple of `base`.
#[inline]
pub fn round_up_to_multiple_of(size: usize, base: usize) -> usize {
    let rem = size % base;
    if rem == 0 {
        size
    } else {
        size + base - rem
    }
}
