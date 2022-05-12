use alloc::alloc::{AllocError, Allocator, Global, Layout};
use alloc::boxed::Box;
use core::cmp;
use core::ops::Range;
use core::ptr::{self, NonNull};

use intrusive_collections::intrusive_adapter;
use intrusive_collections::{LinkedListLink, UnsafeRef};

use liblumen_system::arch::MIN_ALIGN;

use crate::heap::Heap;

// This adapter is used to track a list of heap fragments, attached to a process
intrusive_adapter!(pub HeapFragmentAdapter = UnsafeRef<HeapFragment>: HeapFragment { link: LinkedListLink });

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RawFragment {
    layout: Layout,
    base: NonNull<u8>,
}
impl RawFragment {
    /// Get a pointer to the data in this heap fragment
    #[inline]
    pub fn data(&self) -> NonNull<u8> {
        self.base
    }

    /// Return a pointer range representing the addressable memory of this fragment
    #[inline]
    pub fn as_ptr_range(&self) -> Range<*mut u8> {
        let base = self.base.as_ptr();
        let end = unsafe { base.add(self.layout.size()) };
        base..end
    }

    /// Get the layout of this heap fragment
    #[inline]
    pub fn layout(&self) -> Layout {
        self.layout.clone()
    }
}

pub struct HeapFragment {
    /// Link to the intrusive list that holds all heap fragments
    pub link: LinkedListLink,
    /// The memory region allocated for this fragment
    raw: RawFragment,
    /// A pointer to the top of the allocated region of this fragment,
    /// e.g. when the fragment is unused, `top == raw.base`
    top: *mut u8,
    /// An optional destructor for this fragment
    destructor: Option<Box<dyn Fn(NonNull<u8>)>>,
}
impl HeapFragment {
    /// Returns the pointer to the data region of this fragment
    #[inline]
    pub fn data(&self) -> NonNull<u8> {
        self.raw.data()
    }

    /// Creates a new heap fragment with the given layout, allocated on the global heap
    #[inline]
    pub fn new(
        layout: Layout,
        destructor: Option<Box<dyn Fn(NonNull<u8>)>>,
    ) -> Result<NonNull<Self>, AllocError> {
        let align = cmp::max(MIN_ALIGN, layout.align());
        let layout = layout.align_to(align).unwrap().pad_to_align();

        let (full_layout, offset) = Layout::new::<Self>().extend(layout.clone()).unwrap();
        let ptr: NonNull<u8> = Global.allocate(full_layout)?.cast();
        let header = ptr.as_ptr() as *mut Self;
        let base = unsafe { NonNull::new_unchecked(ptr.as_ptr().add(offset)) };
        unsafe {
            header.write(Self {
                link: LinkedListLink::new(),
                raw: RawFragment { layout, base },
                top: base.as_ptr(),
                destructor,
            });
            Ok(NonNull::new_unchecked(header))
        }
    }
}
impl Drop for HeapFragment {
    fn drop(&mut self) {
        assert!(!self.link.is_linked());
        // Check if this fragment needs to have a destructor run
        if let Some(ref destructor) = self.destructor {
            destructor(self.raw.base);
        }

        // Deallocate the memory backing this fragment
        let (layout, _offset) = Layout::new::<Self>().extend(self.raw.layout()).unwrap();
        unsafe {
            let ptr = NonNull::new_unchecked(self as *const _ as *mut u8);
            Global.deallocate(ptr, layout);
        }
    }
}
unsafe impl Allocator for HeapFragment {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let layout = layout.pad_to_align();
        let size = layout.size();

        // Calculate the base pointer of the allocation at the desired alignment,
        // then offset that pointer by the desired size to give us the new top
        let top = self.top;
        let offset = top.align_offset(layout.align());
        let base = unsafe { top.add(offset) };
        let new_top = unsafe { base.add(size) };

        // Make sure the requested allocation fits within the fragment
        let range = self.raw.as_ptr_range();
        if range.contains(&new_top) {
            Ok(unsafe { NonNull::new_unchecked(ptr::from_raw_parts_mut(base.cast(), size)) })
        } else {
            Err(AllocError)
        }
    }

    // The following functions are all no-ops or errors with heap fragments

    unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: Layout) {}
    unsafe fn grow(
        &self,
        _ptr: NonNull<u8>,
        _old_layout: Layout,
        _new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        Err(AllocError)
    }
    unsafe fn grow_zeroed(
        &self,
        _ptr: NonNull<u8>,
        _old_layout: Layout,
        _new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        Err(AllocError)
    }
    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        _old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        Ok(NonNull::slice_from_raw_parts(ptr, new_layout.size()))
    }
}
impl Heap for HeapFragment {
    #[inline]
    fn heap_start(&self) -> *mut u8 {
        self.raw.base.as_ptr()
    }

    #[inline]
    fn heap_top(&self) -> *mut u8 {
        self.top
    }

    #[inline]
    fn heap_end(&self) -> *mut u8 {
        self.raw.as_ptr_range().end
    }
}
