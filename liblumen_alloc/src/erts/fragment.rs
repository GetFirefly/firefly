use core::alloc::{AllocErr, Layout};
use core::ptr::{self, NonNull};

use intrusive_collections::intrusive_adapter;
use intrusive_collections::{LinkedListLink, UnsafeRef};
use liblumen_core::util::pointer::{distance_absolute, in_area};

use crate::std_alloc;

use super::{HeapAlloc, Term};

// This adapter is used to track a list of heap fragments, attached to a process
intrusive_adapter!(pub HeapFragmentAdapter = UnsafeRef<HeapFragment>: HeapFragment { link: LinkedListLink });

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RawFragment {
    size: usize,
    align: usize,
    data: *mut u8,
}
impl RawFragment {
    /// Returns the size (in bytes) of the fragment
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get a pointer to the data in this heap fragment
    #[inline]
    pub fn data(&self) -> NonNull<u8> {
        unsafe { NonNull::new_unchecked(self.data) }
    }

    /// Get the layout of this heap fragment
    #[inline]
    pub fn layout(&self) -> Layout {
        unsafe { Layout::from_size_align_unchecked(self.size, self.align) }
    }

    /// Returns true if the given pointer is contained within this fragment
    #[inline]
    pub fn contains<T>(&self, ptr: *const T) -> bool {
        let ptr = ptr as usize;
        let start = self.data as usize;
        let end = unsafe { self.data.offset(self.size as isize) } as usize;
        start <= ptr && ptr <= end
    }
}

#[derive(Debug)]
pub struct HeapFragment {
    // Link to the intrusive list that holds all heap fragments
    pub link: LinkedListLink,
    // The memory region allocated for this fragment
    raw: RawFragment,
    // The amount of used memory in this fragment
    top: *mut u8,
}
impl HeapFragment {
    /// Returns the size (in bytes) of the fragment
    #[inline]
    pub fn size(&self) -> usize {
        self.raw.size()
    }

    /// Returns the pointer to the data region of this fragment
    #[inline]
    pub fn data(&self) -> NonNull<u8> {
        self.raw.data()
    }

    /// Returns true if the given pointer is contained within this fragment
    #[inline]
    pub fn contains<T>(&self, ptr: *const T) -> bool {
        self.raw.contains(ptr)
    }

    /// Creates a new heap fragment with the given layout, allocated via `std_alloc`
    #[inline]
    pub unsafe fn new(layout: Layout) -> Result<NonNull<Self>, AllocErr> {
        let (full_layout, offset) = Layout::new::<Self>().extend(layout.clone()).unwrap();
        let size = layout.size();
        let align = layout.align();
        let ptr = std_alloc::alloc(full_layout)?.as_ptr() as *mut Self;
        let data = (ptr as *mut u8).offset(offset as isize);
        let top = data;
        ptr::write(
            ptr,
            Self {
                link: LinkedListLink::new(),
                raw: RawFragment { size, align, data },
                top,
            },
        );
        Ok(NonNull::new_unchecked(ptr))
    }
}
impl Drop for HeapFragment {
    fn drop(&mut self) {
        assert!(!self.link.is_linked());
        // Check if the contained value needs to have its destructor run
        let ptr = self.data().as_ptr() as *mut Term;
        let term = unsafe { *ptr };
        term.release();
        // Actually deallocate the memory backing this fragment
        let (layout, _offset) = Layout::new::<Self>().extend(self.raw.layout()).unwrap();
        unsafe {
            let ptr = NonNull::new_unchecked(self as *const _ as *mut u8);
            std_alloc::dealloc(ptr, layout);
        }
    }
}
impl HeapAlloc for HeapFragment {
    /// Perform a heap allocation.
    ///
    /// If space on the process heap is not immediately available, then the allocation
    /// will be pushed into a heap fragment which will then be later moved on to the
    /// process heap during garbage collection
    unsafe fn alloc(&mut self, need: usize) -> Result<NonNull<Term>, AllocErr> {
        let base = self.raw.data as *mut Term;
        let top = self.top as *mut Term;
        let available = distance_absolute(top, base);
        if need > available {
            return Err(AllocErr);
        }

        let new_top = base.offset(need as isize);
        debug_assert!(new_top <= self.raw.data.offset(self.raw.size as isize) as *mut Term);
        self.top = new_top as *mut u8;
        Ok(NonNull::new_unchecked(new_top))
    }

    /// Returns true if the given pointer is owned by this process/heap
    fn is_owner<T>(&mut self, ptr: *const T) -> bool {
        in_area(ptr, self.raw.data, self.top)
    }
}
