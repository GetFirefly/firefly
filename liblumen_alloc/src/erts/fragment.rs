use core::mem;
use core::ptr::{self, NonNull};

use intrusive_collections::intrusive_adapter;
use intrusive_collections::{LinkedListLink, UnsafeRef};

use liblumen_core::alloc::prelude::*;
use liblumen_core::alloc::utils::{align_up_to, is_aligned, is_aligned_at};

use crate::erts;
use crate::erts::exception::AllocResult;
use crate::erts::process::alloc::{Heap, HeapAlloc};
use crate::erts::term::prelude::*;
use crate::std_alloc;
use liblumen_core::sys::sysconf::MIN_ALIGN;

// This adapter is used to track a list of heap fragments, attached to a process
intrusive_adapter!(pub HeapFragmentAdapter = UnsafeRef<HeapFragment>: HeapFragment { link: LinkedListLink });

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RawFragment {
    size: usize,
    align: usize,
    base: NonNull<u8>,
}
impl RawFragment {
    /// Get a pointer to the data in this heap fragment
    #[inline]
    pub fn data(&self) -> NonNull<u8> {
        self.base
    }

    /// Get the layout of this heap fragment
    #[inline]
    pub fn layout(&self) -> Layout {
        unsafe { Layout::from_size_align_unchecked(self.size, self.align) }
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
    /// Returns the pointer to the data region of this fragment
    #[inline]
    pub fn data(&self) -> NonNull<u8> {
        self.raw.data()
    }

    /// Creates a new heap fragment with the given layout, allocated via `std_alloc`
    #[inline]
    pub fn new(layout: Layout) -> AllocResult<NonNull<Self>> {
        // `alloc_layout` pads to `MIN_ALIGN`, so creating the new `HeapFragment` must too
        // Ensure layout has alignment padding
        let layout = layout.align_to(MIN_ALIGN).unwrap().pad_to_align();
        let (full_layout, offset) = Layout::new::<Self>().extend(layout.clone()).unwrap();
        let size = layout.size();
        let align = layout.align();
        let block = std_alloc::alloc(full_layout, AllocInit::Uninitialized)?;
        let ptr = block.ptr.as_ptr() as *mut Self;
        let data = unsafe { (ptr as *mut u8).add(offset) };
        let top = data;
        unsafe {
            ptr::write(
                ptr,
                Self {
                    link: LinkedListLink::new(),
                    raw: RawFragment {
                        size,
                        align,
                        base: NonNull::new_unchecked(data),
                    },
                    top,
                },
            );
        }
        Ok(block.ptr.cast())
    }

    pub fn new_from_word_size(word_size: usize) -> AllocResult<NonNull<Self>> {
        let byte_size = word_size * mem::size_of::<Term>();
        let align = mem::align_of::<Term>();

        let layout = unsafe { Layout::from_size_align_unchecked(byte_size, align) };

        Self::new(layout)
    }
}
impl Drop for HeapFragment {
    fn drop(&mut self) {
        assert!(!self.link.is_linked());
        // Check if the contained value needs to have its destructor run
        let ptr = self.raw.base.as_ptr() as *mut Term;
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
impl Heap for HeapFragment {
    #[inline]
    fn heap_start(&self) -> *mut Term {
        self.raw.base.as_ptr() as *mut Term
    }

    #[inline]
    fn heap_top(&self) -> *mut Term {
        self.top as *mut Term
    }

    #[inline]
    fn heap_end(&self) -> *mut Term {
        unsafe { self.raw.base.as_ptr().add(self.raw.size) as *mut Term }
    }
}
impl HeapAlloc for HeapFragment {
    unsafe fn alloc_layout(&mut self, layout: Layout) -> AllocResult<NonNull<Term>> {
        // Ensure layout has alignment padding
        let layout = layout.align_to(MIN_ALIGN).unwrap().pad_to_align();
        // Capture the base pointer for this allocation
        let top = self.heap_top() as *mut u8;
        // Calculate available space and fail if not enough is free
        let needed = layout.size();
        let end = self.heap_end() as *mut u8;
        if erts::to_word_size(needed) > self.heap_available() {
            return Err(alloc!());
        }
        // Calculate new top of the heap
        let new_top = top.add(needed);
        debug_assert!(new_top <= end);
        self.top = new_top;
        // Ensure base pointer for allocation fulfills minimum alignment requirements
        let align = layout.align();
        let ptr = if is_aligned_at(top, align) {
            top as *mut Term
        } else {
            align_up_to(top as *mut Term, align)
        };
        // Success!
        debug_assert!(is_aligned(ptr));
        Ok(NonNull::new_unchecked(ptr))
    }
}
