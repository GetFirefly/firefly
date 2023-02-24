use alloc::alloc::{AllocError, Allocator, Global, Layout};
use core::cell::UnsafeCell;
use core::mem;
use core::ptr::{self, NonNull};

use firefly_alloc::heap::{Heap, HeapMut};

use crate::term::OpaqueTerm;

pub struct ProcessHeap {
    range: *mut [u8],
    top: UnsafeCell<*mut u8>,
    high_water_mark: *mut u8,
}
impl ProcessHeap {
    const SIZES: [usize; 22] = generate_sizes::<22>();
    pub const DEFAULT_SIZE: usize = Self::SIZES[0];

    pub fn new(size: usize) -> Self {
        let layout = Layout::from_size_align(size, mem::align_of::<OpaqueTerm>()).unwrap();
        let nonnull = Global.allocate(layout).unwrap();
        let top = nonnull.as_non_null_ptr().as_ptr();
        Self {
            range: nonnull.as_ptr(),
            top: UnsafeCell::new(top),
            high_water_mark: top,
        }
    }

    pub fn empty() -> Self {
        Self {
            range: ptr::from_raw_parts_mut(ptr::null_mut(), 0),
            top: UnsafeCell::new(ptr::null_mut()),
            high_water_mark: ptr::null_mut(),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.range.len() == 0
    }

    pub fn next_size(size: usize) -> usize {
        match Self::SIZES.binary_search(&size) {
            Ok(i) | Err(i) => {
                if let Some(size) = Self::SIZES.get(i + 1) {
                    *size
                } else {
                    let next_size = (size as f64 * 1.2) as usize;
                    next_size.next_multiple_of(16)
                }
            }
        }
    }
}

const fn generate_sizes<const N: usize>() -> [usize; N] {
    let mut sizes = [0; N];
    let mut k = 232;
    let mut i = 0;
    while i < N {
        k = next_size(k);
        sizes[i] = k;
        i += 1;
    }
    sizes
}

const fn next_size(k: usize) -> usize {
    let mut a = 1;
    let mut b = 1;
    while b <= k {
        let (c, d) = (a, b);
        a = b;
        b = c + d;
    }
    b
}

impl Default for ProcessHeap {
    #[inline]
    fn default() -> Self {
        Self::new(Self::DEFAULT_SIZE)
    }
}
impl Drop for ProcessHeap {
    fn drop(&mut self) {
        let size = ptr::metadata(self.range) as usize;
        let layout = Layout::from_size_align(size, mem::align_of::<OpaqueTerm>()).unwrap();
        unsafe { Global.deallocate(NonNull::new_unchecked(self.range.cast()), layout) }
    }
}
unsafe impl Allocator for ProcessHeap {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let layout = layout.pad_to_align();
        let size = layout.size();

        // Calculate the base pointer of the allocation at the desired alignment,
        // then offset that pointer by the desired size to give us the new top
        let top = unsafe { *self.top.get() };
        let offset = top.align_offset(layout.align());
        let base = unsafe { top.add(offset) };
        let new_top = unsafe { base.add(size) } as *const u8;

        // Make sure the requested allocation fits within the fragment
        let start = self.range.as_mut_ptr() as *const u8;
        let heap_size = self.range.len();
        let range = start..(unsafe { start.add(heap_size) });
        if range.contains(&new_top) {
            unsafe {
                self.top.get().write(new_top as *mut u8);
            }
            Ok(unsafe { NonNull::new_unchecked(ptr::from_raw_parts_mut(base.cast(), size)) })
        } else {
            Err(AllocError)
        }
    }

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
impl Heap for ProcessHeap {
    #[inline]
    fn heap_start(&self) -> *mut u8 {
        self.range.as_mut_ptr()
    }

    #[inline]
    fn heap_top(&self) -> *mut u8 {
        unsafe { *self.top.get() }
    }

    #[inline]
    unsafe fn reset_heap_top(&self, ptr: *mut u8) {
        self.top.get().write(ptr);
    }

    #[inline]
    fn heap_end(&self) -> *mut u8 {
        unsafe { self.heap_start().add(self.range.len()) }
    }

    #[inline]
    fn high_water_mark(&self) -> Option<NonNull<u8>> {
        NonNull::new(self.high_water_mark)
    }
}
impl HeapMut for ProcessHeap {
    #[inline]
    fn set_high_water_mark(&mut self, ptr: *mut u8) {
        assert!(self.contains(ptr.cast()));
        assert!(ptr <= self.heap_top());
        self.high_water_mark = ptr;
    }
}
