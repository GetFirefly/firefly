#[cfg(has_mmap)]
mod internal {
    use core::ptr::{self, NonNull};
    use core::alloc::{Layout, AllocErr};
    use crate::sys::mmap;

    #[inline]
    pub unsafe fn map(layout: Layout) -> Result<NonNull<u8>, AllocErr> {
        mmap::map(ptr::null_mut(), layout.size())
    }

    #[inline]
    pub unsafe fn unmap(ptr: *mut u8, layout: Layout) {
        mmap::unmap(ptr, layout.size());
    }

    #[allow(unused)]
    #[inline]
    pub unsafe fn uncommit(ptr: *mut u8, size: usize) {
        mmap::discard(ptr, size);
    }
}

#[cfg(not(has_mmap))]
mod internal {
    use core::ptr::{self, NonNull};
    use core::alloc::{Layout, AllocErr};
    use crate::sys;

    #[inline]
    pub unsafe fn map(layout: Layout) -> Result<NonNull<u8>, AllocErr> {
        sys::alloc(layout)
    }

    #[inline]
    pub unsafe fn unmap(ptr: *mut u8, layout: Layout) {
        sys::free(ptr, layout);
    }

    // NOTE: This is here just to keep things symmetrical, but it is unused
    // as we don't try to free a subset of an allocation from sys_alloc
    #[allow(unused)]
    #[inline]
    pub unsafe fn uncommit(ptr: *mut u8, size: usize) {}
}

pub use self::internal::{
    map,
    unmap,
    uncommit,
};
