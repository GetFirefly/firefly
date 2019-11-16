use core::alloc::Layout;
use core::fmt;
use core::mem;
use core::ptr::NonNull;

use liblumen_core::alloc::utils::{align_up_to, is_aligned, is_aligned_at};
use liblumen_core::sys::alloc as sys_alloc;
use liblumen_core::sys::sysconf::MIN_ALIGN;
use liblumen_core::util::pointer::distance_absolute;

use crate::erts::exception::AllocResult;
use crate::erts::process::alloc::{
    Heap, HeapAlloc, VirtualAllocator, VirtualBinaryHeap, VirtualHeap,
};
use crate::erts::term::prelude::*;
use crate::mem::bit_size_of;

// We allocate 64 words for default scratch heaps, which amounts
// to 512 bytes for 64-bit systems, and 256 for 32-bit systems;
// this amount should provide enough working room for just about
// any unit test allocation - for larger requirements, manually
// define a layout and use `RegionHeap::new`
pub(in crate::erts) const DEFAULT_HEAP_SIZE: usize = mem::size_of::<Term>() * 32;

/// This struct defines a system allocator-backed heap
/// for testing functionality which requires a `HeapAlloc`
/// parameter. It releases allocated memory when dropped.
pub struct RegionHeap {
    layout: Layout,
    ptr: NonNull<u8>,
    end: *mut u8,
    top: *mut u8,
    high_water_mark: *mut u8,
    vheap: VirtualBinaryHeap,
}
impl RegionHeap {
    /// Creates a new scratch heap from the given layout
    pub fn new(layout: Layout) -> Self {
        let size = layout.size();
        let ptr =
            unsafe { sys_alloc::alloc(layout.clone()).expect("unable to allocate scratch heap!") };
        let raw = ptr.as_ptr();
        let end = unsafe { raw.add(size) };
        let top = raw;
        Self {
            layout,
            ptr,
            end,
            top,
            high_water_mark: top,
            vheap: VirtualBinaryHeap::new(size),
        }
    }

    /// Sets the high water mark to the current top of the heap
    #[inline]
    pub fn set_high_water_mark(&mut self) {
        self.high_water_mark = self.top;
    }
}
impl Default for RegionHeap {
    fn default() -> Self {
        // Allocate enough space for most tests
        let layout = Layout::from_size_align(DEFAULT_HEAP_SIZE, MIN_ALIGN)
            .expect("invalid size/alignment for DEFAULT_HEAP_SIZE");
        Self::new(layout)
    }
}
impl VirtualHeap<ProcBin> for RegionHeap {
    #[inline]
    fn virtual_size(&self) -> usize {
        self.vheap.virtual_size()
    }

    #[inline]
    fn virtual_heap_used(&self) -> usize {
        self.vheap.virtual_heap_used()
    }

    #[inline]
    fn virtual_heap_unused(&self) -> usize {
        self.vheap.virtual_heap_unused()
    }
}
impl VirtualAllocator<ProcBin> for RegionHeap {
    #[inline]
    fn virtual_alloc(&mut self, value: Boxed<ProcBin>) {
        self.vheap.virtual_alloc(value);
    }

    #[inline]
    fn virtual_free(&mut self, value: Boxed<ProcBin>) {
        self.vheap.virtual_free(value);
    }

    #[inline]
    fn virtual_unlink(&mut self, value: Boxed<ProcBin>) {
        self.vheap.virtual_unlink(value);
    }

    #[inline]
    fn virtual_pop(&mut self, value: Boxed<ProcBin>) -> ProcBin {
        self.vheap.virtual_pop(value)
    }

    #[inline]
    fn virtual_contains<P: ?Sized>(&self, ptr: *const P) -> bool {
        self.vheap.virtual_contains(ptr)
    }

    #[inline]
    unsafe fn virtual_clear(&mut self) {
        self.vheap.virtual_clear();
    }
}
impl Heap for RegionHeap {
    #[inline]
    fn heap_start(&self) -> *mut Term {
        self.ptr.as_ptr() as *mut Term
    }

    #[inline]
    fn heap_top(&self) -> *mut Term {
        self.top as *mut Term
    }

    #[inline]
    fn heap_end(&self) -> *mut Term {
        self.end as *mut Term
    }

    #[inline]
    fn high_water_mark(&self) -> *mut Term {
        self.high_water_mark as *mut Term
    }
}
impl HeapAlloc for RegionHeap {
    /// Perform a heap allocation.
    ///
    /// If space on the process heap is not immediately available, then the allocation
    /// will be pushed into a heap fragment which will then be later moved on to the
    /// process heap during garbage collection
    unsafe fn alloc_layout(&mut self, layout: Layout) -> AllocResult<NonNull<Term>> {
        // Ensure layout has alignment padding
        let layout = layout.align_to(MIN_ALIGN).unwrap().pad_to_align().unwrap();
        // Capture the base pointer for this allocation
        let top = self.top;
        // Calculate available space and fail if not enough is free
        let needed = layout.size();
        let available = distance_absolute(self.end, top);
        if needed > available {
            return Err(alloc!());
        }
        // Calculate new top of the heap
        let new_top = top.add(needed);
        debug_assert!(new_top <= self.end);
        self.top = new_top;
        // Ensure base pointer for allocation fulfills minimum alignment requirements
        let align = layout.align();
        let ptr = if is_aligned_at(top, align) {
            top as *mut Term
        } else {
            align_up_to(top as *mut Term, align)
        };
        // Success!
        debug_assert!(
            is_aligned(ptr),
            "unaligned pointer ({:b}); requested alignment is {}",
            ptr as usize,
            align
        );
        Ok(NonNull::new_unchecked(ptr))
    }
}
impl fmt::Debug for RegionHeap {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_fmt(format_args!(
            "RegionHeap (size: {}, align: {}, heap: {:p}-{:p}, used: {}, unused: {}) [\n",
            self.layout.size(),
            self.layout.align(),
            self.heap_start(),
            self.heap_top(),
            self.heap_used(),
            self.heap_available(),
        ))?;
        let mut pos = self.heap_start();
        while pos < self.heap_top() {
            unsafe {
                let term = &*pos;
                let skip = term.arity();
                f.write_fmt(format_args!(
                    "  {:p}: {:0bit_len$b} {:?}\n",
                    pos,
                    *(pos as *const usize),
                    term,
                    bit_len = (bit_size_of::<usize>())
                ))?;
                pos = pos.add(1 + skip);
            }
        }
        f.write_fmt(format_args!("  {:p}: END OF HEAP", pos))?;
        f.write_str("]\n")
    }
}
impl Drop for RegionHeap {
    fn drop(&mut self) {
        unsafe {
            sys_alloc::free(self.ptr.as_ptr(), self.layout.clone());
        }
    }
}
