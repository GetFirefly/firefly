use core::alloc::{AllocError, Layout};
use core::ptr;

use liblumen_alloc::mmap;
use liblumen_system as system;

const STACK_ALIGNMENT: usize = 16;

#[derive(Debug)]
pub struct ProcessStack {
    pub base: *mut u8,
    pub top: *mut u8,
    pub size: usize,
    pub bottom: *mut u8,
}
impl ProcessStack {
    /// Allocate a new process stack of the given size (in pages)
    pub fn new(num_pages: usize) -> Result<Self, AllocError> {
        debug_assert!(num_pages > 0, "stack size in pages must be greater than 0");

        let ptr = unsafe { mmap::map_stack(num_pages)? };
        Ok(unsafe { Self::from_raw_parts(ptr.as_ptr(), num_pages) })
    }

    unsafe fn from_raw_parts(base: *mut u8, pages: usize) -> Self {
        let page_size = system::arch::page_size();
        let size = (pages + 1) * page_size;

        // The top of the stack is where growth starts (stack grows downwards towards bottom)
        let top = base.add(size);
        // The bottom is where the usable space ends and where the guard page begins
        let bottom = base.add(page_size);
        assert_eq!(
            bottom as usize % STACK_ALIGNMENT,
            0,
            "expected allocated stack to meet minimum alignment requirements"
        );

        Self {
            base,
            top,
            size,
            bottom,
        }
    }

    #[inline]
    pub fn limit(&self) -> *mut u8 {
        self.bottom
    }

    #[inline]
    pub fn is_guard_page<T>(&self, addr: *mut T) -> bool {
        system::mem::in_area_inclusive(addr, self.base, self.bottom)
    }
}
impl Default for ProcessStack {
    fn default() -> Self {
        Self {
            base: ptr::null_mut(),
            top: ptr::null_mut(),
            size: 0,
            bottom: ptr::null_mut(),
        }
    }
}
impl Drop for ProcessStack {
    fn drop(&mut self) {
        if self.base.is_null() {
            return;
        }

        let page_size = system::arch::page_size();
        let pages = (self.size / page_size) - 1;

        let (layout, _offset) = Layout::from_size_align(page_size, page_size)
            .unwrap()
            .repeat(pages)
            .unwrap();

        unsafe {
            mmap::unmap(self.base, layout);
        }
    }
}
