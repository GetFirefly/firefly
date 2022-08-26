// This module provides the fallback implementation of mmap primitives on platforms which do not provide them
#[cfg(any(target_arch = "wasm32", target_arch = "wasm64"))]
mod fallback {
    use alloc::alloc::{AllocError, Layout};
    use core::ptr::NonNull;

    use firefly_system::arch as sys;

    /// Creates a memory mapping for the given `Layout`
    #[inline]
    pub unsafe fn map(layout: Layout) -> Result<NonNull<u8>, AllocError> {
        sys::alloc::allocate(layout).map(|ptr| ptr.cast())
    }

    /// Creates a memory mapping specifically set up to behave like a stack
    ///
    /// NOTE: This is a fallback implementation, so no guard page is present,
    /// and it is implemented on top of plain `map`
    #[inline]
    pub unsafe fn map_stack(pages: usize) -> Result<NonNull<u8>, AllocError> {
        let page_size = sys::mem::page_size();
        let layout = Layout::from_size_align(page_size * pages, page_size).unwrap();

        sys::alloc::allocate(layout).map(|ptr| ptr.cast())
    }

    /// Remaps a mapping given a pointer to the mapping, the layout which created it, and the new size
    #[inline]
    pub unsafe fn remap(
        ptr: *mut u8,
        old_layout: Layout,
        new_size: usize,
    ) -> Result<NonNull<u8>, AllocError> {
        let new_layout =
            Layout::from_size_align(new_size, old_layout.align()).map_err(|_| AllocError)?;
        let ptr = NonNull::new(ptr).ok_or(AllocError)?;

        if old_layout.size() < new_size {
            sys::alloc::grow(ptr, old_layout, new_layout)
        } else {
            sys::alloc::shrink(ptr, old_layout, new_layout)
        }
        .map(|ptr| ptr.cast())
    }

    /// Destroys a mapping given a pointer to the mapping and the layout which created it
    #[inline]
    pub unsafe fn unmap(ptr: *mut u8, layout: Layout) {
        if ptr.is_null() {
            return;
        }
        sys::alloc::deallocate(NonNull::new_unchecked(ptr), layout);
    }
}

// This module provides the real implementation of mmap primitives on platforms which provide them
#[cfg(not(any(target_arch = "wasm32", target_arch = "wasm64")))]
mod real {
    use alloc::alloc::{AllocError, Layout};
    use core::ptr::NonNull;

    use firefly_system::arch as sys;

    /// Creates a memory mapping for the given `Layout`
    #[inline]
    pub unsafe fn map(layout: Layout) -> Result<NonNull<u8>, AllocError> {
        sys::mmap::map(layout).map(|(ptr, _)| ptr)
    }

    /// Creates a memory mapping specifically set up to behave like a stack
    #[inline]
    pub unsafe fn map_stack(pages: usize) -> Result<NonNull<u8>, AllocError> {
        sys::mmap::map_stack(pages)
    }

    /// Remaps a mapping given a pointer to the mapping, the layout which created it, and the new size
    #[inline]
    pub unsafe fn remap(
        ptr: *mut u8,
        layout: Layout,
        new_size: usize,
    ) -> Result<NonNull<u8>, AllocError> {
        sys::mmap::remap(ptr, layout, new_size).map(|(ptr, _)| ptr)
    }

    /// Destroys a mapping given a pointer to the mapping and the layout which created it
    #[inline]
    pub unsafe fn unmap(ptr: *mut u8, layout: Layout) {
        sys::mmap::unmap(ptr, layout);
    }
}

#[cfg(any(target_arch = "wasm32", target_arch = "wasm64"))]
pub use self::fallback::*;

#[cfg(not(any(target_arch = "wasm32", target_arch = "wasm64")))]
pub use self::real::*;
