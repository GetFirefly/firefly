use core::alloc::Layout;
use core::ptr::NonNull;
use core::mem;

use liblumen_core::sys::alloc as sys_alloc;
use liblumen_core::sys::sysconf::MIN_ALIGN;
use liblumen_core::util::pointer::{distance_absolute, in_area};
use liblumen_core::alloc::utils::{is_aligned, is_aligned_at, align_up_to};

use crate::erts::term::prelude::{Term, ProcBin};
use crate::erts::exception::AllocResult;
use crate::erts::process::alloc::{HeapAlloc, VirtualAlloc};
use crate::erts::process::VirtualBinaryHeap;

// We allocate 64 words for default scratch heaps, which amounts
// to 512 bytes for 64-bit systems, and 256 for 32-bit systems;
// this amount should provide enough working room for just about
// any unit test allocation - for larger requirements, manually
// define a layout and use `RegionHeap::new`
const DEFAULT_HEAP_SIZE: usize = mem::size_of::<Term>() * 32;

/// This struct defines a system allocator-backed heap
/// for testing functionality which requires a `HeapAlloc`
/// parameter. It releases allocated memory when dropped.
pub struct RegionHeap {
    layout: Layout,
    ptr: NonNull<u8>,
    end: *mut u8,
    top: *mut u8,
    vheap: VirtualBinaryHeap,
}
impl RegionHeap {
    /// Creates a new scratch heap from the given layout
    pub fn new(layout: Layout) -> Self {
        let size = layout.size();
        let ptr = unsafe {
            sys_alloc::alloc(layout.clone())
                .expect("unable to allocate scratch heap!")
        };
        let raw = ptr.as_ptr();
        let end = unsafe { raw.add(size) };
        let top = raw;
        Self {
            layout,
            ptr,
            end,
            top,
            vheap: VirtualBinaryHeap::new(size),
        }
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
impl VirtualAlloc for RegionHeap {
    #[inline]
    fn virtual_alloc(&mut self, bin: &ProcBin) -> Term {
        self.vheap.push(bin)
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
        let layout = layout.pad_to_align().unwrap();
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
        debug_assert!(is_aligned(ptr), "unaligned pointer ({:b}); requested alignment is {}", ptr as usize, align);
        Ok(NonNull::new_unchecked(ptr))
    }

    /// Returns true if the given pointer is owned by this process/heap
    fn is_owner<T>(&mut self, ptr: *const T) -> bool where T: ?Sized {
        in_area(ptr, self.ptr.as_ptr(), self.top)
    }
}
impl Drop for RegionHeap {
    fn drop(&mut self) {
        unsafe {
            sys_alloc::free(self.ptr.as_ptr(), self.layout.clone());
        }
    }
}
