pub mod alloc_handle;
pub mod boxed;
pub mod mmap;
pub mod raw_vec;
mod region;
pub mod size_classes;
mod static_alloc;
mod sys_alloc;
pub mod utils;
pub mod vec;

#[rustversion::before(2020-01-30)]
use core::{
    cmp,
    ptr::{self, NonNull},
};

pub use self::region::Region;
pub use self::static_alloc::StaticAlloc;
pub use self::sys_alloc::*;

// Re-export core alloc types
pub use core::alloc::{AllocErr, CannotReallocInPlace, GlobalAlloc, Layout, LayoutErr};

// Smooth transition to AllocRef
#[rustversion::since(2020-01-30)]
pub use core::alloc::AllocRef;
#[rustversion::before(2020-01-30)]
pub unsafe trait AllocRef {
    unsafe fn alloc(&mut self, layout: Layout) -> Result<(NonNull<u8>, usize), AllocErr>;
    unsafe fn dealloc(&mut self, ptr: NonNull<u8>, layout: Layout);
    unsafe fn alloc_zeroed(&mut self, layout: Layout) -> Result<(NonNull<u8>, usize), AllocErr> {
        let size = layout.size();
        let result = self.alloc(layout);
        if let Ok((p, _)) = result {
            ptr::write_bytes(p.as_ptr(), 0, size);
        }
        result
    }

    unsafe fn realloc(
        &mut self,
        ptr: NonNull<u8>,
        layout: Layout,
        new_size: usize,
    ) -> Result<(NonNull<u8>, usize), AllocErr> {
        let old_size = layout.size();

        if new_size > old_size {
            if let Ok(size) = self.grow_in_place(ptr, layout, new_size) {
                return Ok((ptr, size));
            }
        } else if new_size < old_size {
            if let Ok(size) = self.shrink_in_place(ptr, layout, new_size) {
                return Ok((ptr, size));
            }
        } else {
            return Ok((ptr, new_size));
        }

        // otherwise, fall back on alloc + copy + dealloc.
        let new_layout = Layout::from_size_align_unchecked(new_size, layout.align());
        let result = self.alloc(new_layout);
        if let Ok((new_ptr, _)) = result {
            ptr::copy_nonoverlapping(ptr.as_ptr(), new_ptr.as_ptr(), cmp::min(old_size, new_size));
            self.dealloc(ptr, layout);
        }
        result
    }

    unsafe fn realloc_zeroed(
        &mut self,
        ptr: NonNull<u8>,
        layout: Layout,
        new_size: usize,
    ) -> Result<(NonNull<u8>, usize), AllocErr> {
        let old_size = layout.size();

        if new_size > old_size {
            if let Ok(size) = self.grow_in_place_zeroed(ptr, layout, new_size) {
                return Ok((ptr, size));
            }
        } else if new_size < old_size {
            if let Ok(size) = self.shrink_in_place(ptr, layout, new_size) {
                return Ok((ptr, size));
            }
        } else {
            return Ok((ptr, new_size));
        }

        // otherwise, fall back on alloc + copy + dealloc.
        let new_layout = Layout::from_size_align_unchecked(new_size, layout.align());
        let result = self.alloc_zeroed(new_layout);
        if let Ok((new_ptr, _)) = result {
            ptr::copy_nonoverlapping(ptr.as_ptr(), new_ptr.as_ptr(), cmp::min(old_size, new_size));
            self.dealloc(ptr, layout);
        }
        result
    }

    #[inline]
    unsafe fn grow_in_place(
        &mut self,
        ptr: NonNull<u8>,
        layout: Layout,
        new_size: usize,
    ) -> Result<usize, CannotReallocInPlace> {
        let _ = ptr;
        let _ = layout;
        let _ = new_size;
        Err(CannotReallocInPlace)
    }

    unsafe fn grow_in_place_zeroed(
        &mut self,
        ptr: NonNull<u8>,
        layout: Layout,
        new_size: usize,
    ) -> Result<usize, CannotReallocInPlace> {
        let size = self.grow_in_place(ptr, layout, new_size)?;
        ptr.as_ptr()
            .add(layout.size())
            .write_bytes(0, new_size - layout.size());
        Ok(size)
    }

    #[inline]
    unsafe fn shrink_in_place(
        &mut self,
        ptr: NonNull<u8>,
        layout: Layout,
        new_size: usize,
    ) -> Result<usize, CannotReallocInPlace> {
        let _ = ptr;
        let _ = layout;
        let _ = new_size;
        Err(CannotReallocInPlace)
    }
}

#[rustversion::before(2019-01-30)]
impl<T: AllocRef> core::alloc::Alloc for T {
    #[inline]
    unsafe fn alloc(&mut self, layout: Layout) -> Result<NonNull<u8>, AllocErr> {
        <Self as AllocRef>::alloc(self, layout).map(|(ptr, _)| ptr)
    }

    #[inline]
    unsafe fn dealloc(&mut self, ptr: NonNull<u8>, layout: Layout) {
        <Self as AllocRef>::dealloc(self, ptr, layout)
    }
}
