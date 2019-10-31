use core::alloc::Layout;
use core::ptr::{self, NonNull};
use core::mem;

use intrusive_collections::intrusive_adapter;
use intrusive_collections::{LinkedListLink, UnsafeRef};

use liblumen_core::util::pointer::{distance_absolute, in_area};
use liblumen_core::alloc::utils::{is_aligned, is_aligned_at, align_up_to};

use crate::erts::exception::AllocResult;
use crate::erts::term::prelude::*;
use crate::std_alloc;

use super::HeapAlloc;

// This adapter is used to track a list of heap fragments, attached to a process
intrusive_adapter!(pub HeapFragmentAdapter = UnsafeRef<HeapFragment>: HeapFragment { link: LinkedListLink });

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RawFragment {
    size: usize,
    align: usize,
    base: *mut u8,
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
        unsafe { NonNull::new_unchecked(self.base) }
    }

    /// Get the layout of this heap fragment
    #[inline]
    pub fn layout(&self) -> Layout {
        unsafe { Layout::from_size_align_unchecked(self.size, self.align) }
    }

    /// Returns true if the given pointer is contained within this fragment
    #[inline]
    pub fn contains<T>(&self, ptr: *const T) -> bool where T: ?Sized {
        let end = unsafe { self.base.add(self.size) } as *const ();
        in_area(ptr as *const (), self.base as *const (), end)
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
    pub fn contains<T>(&self, ptr: *const T) -> bool where T: ?Sized {
        self.raw.contains(ptr)
    }

    /// Creates a new heap fragment with the given layout, allocated via `std_alloc`
    #[inline]
    pub unsafe fn new(layout: Layout) -> AllocResult<NonNull<Self>> {
        let (full_layout, offset) = Layout::new::<Self>().extend(layout.clone()).unwrap();
        let size = layout.size();
        let align = layout.align();
        let ptr = std_alloc::alloc(full_layout)?.as_ptr() as *mut Self;
        let data = (ptr as *mut u8).add(offset);
        let top = data;
        ptr::write(
            ptr,
            Self {
                link: LinkedListLink::new(),
                raw: RawFragment { size, align, base: data },
                top,
            },
        );
        Ok(NonNull::new_unchecked(ptr))
    }

    pub unsafe fn new_from_word_size(word_size: usize) -> AllocResult<NonNull<Self>> {
        let byte_size = word_size * mem::size_of::<Term>();
        let align = mem::align_of::<Term>();

        let layout = Layout::from_size_align_unchecked(byte_size, align);

        Self::new(layout)
    }

    /// Creates a new `HeapFragment` that can hold a tuple
    pub fn tuple_from_slice(elements: &[Term]) -> AllocResult<(Term, NonNull<HeapFragment>)> {
        // Make sure we have a fragment of the appropriate size
        let mut heap_fragment_box = unsafe {
            let (layout, _) = Tuple::layout_for(elements);
            Self::new(layout)?
        };
        let heap_fragment_ref = unsafe { heap_fragment_box.as_mut() };

        // Then allocate the new tuple in the fragment using the provided elements
        let ptr = Tuple::from_slice(heap_fragment_ref, elements)?;
        // Encode the tuple pointer into a box
        let term = ptr.as_ptr().into();

        Ok((term, heap_fragment_box))
    }
}
impl Drop for HeapFragment {
    fn drop(&mut self) {
        assert!(!self.link.is_linked());
        // Check if the contained value needs to have its destructor run
        let ptr = self.raw.base as *mut Term;
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
    unsafe fn alloc_layout(&mut self, layout: Layout) -> AllocResult<NonNull<Term>> {
        // Ensure layout has alignment padding
        let layout = layout.pad_to_align().unwrap();
        // Capture the base pointer for this allocation
        let top = self.top;
        // Calculate available space and fail if not enough is free
        let needed = layout.size();
        let end = self.raw.base.add(self.raw.size);
        let available = distance_absolute(end, top);
        if needed > available {
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

    /// Returns true if the given pointer is owned by this process/heap
    fn is_owner<T>(&mut self, ptr: *const T) -> bool where T: ?Sized {
        self.contains(ptr)
    }
}
