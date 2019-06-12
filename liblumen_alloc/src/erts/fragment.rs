use core::alloc::{AllocErr, Layout};
use core::ptr::{self, NonNull};

use intrusive_collections::LinkedListLink;

use crate::std_alloc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RawFragment {
    size: usize,
    align: usize,
    data: *mut u8,
}
impl RawFragment {
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
}

#[derive(Debug)]
pub struct HeapFragment {
    // Link to the intrusive list that holds all heap fragments
    pub link: LinkedListLink,
    // The memory region allocated for this fragment
    raw: RawFragment,
}
impl HeapFragment {
    #[inline]
    pub fn data(&self) -> NonNull<u8> {
        self.raw.data()
    }

    #[inline]
    pub unsafe fn new(layout: Layout) -> Result<NonNull<Self>, AllocErr> {
        let (full_layout, offset) = Layout::new::<Self>().extend(layout.clone()).unwrap();
        let size = layout.size();
        let align = layout.align();
        let ptr = std_alloc::alloc(full_layout)?.as_ptr() as *mut Self;
        let data = (ptr as *mut u8).offset(offset as isize);
        ptr::write(
            ptr,
            Self {
                link: LinkedListLink::new(),
                raw: RawFragment { size, align, data },
            },
        );
        Ok(NonNull::new_unchecked(ptr))
    }
}
impl Drop for HeapFragment {
    fn drop(&mut self) {
        assert!(!self.link.is_linked());
        let (layout, _offset) = Layout::new::<Self>().extend(self.raw.layout()).unwrap();
        unsafe {
            let ptr = NonNull::new_unchecked(self as *const _ as *mut u8);
            std_alloc::dealloc(ptr, layout);
        }
    }
}
