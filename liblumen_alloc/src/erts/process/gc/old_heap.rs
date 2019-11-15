use core::alloc::Layout;
use core::fmt;
use core::mem;
use core::ptr::{self, NonNull};

use liblumen_core::alloc::utils::{align_up_to, is_aligned, is_aligned_at};

use crate::erts::exception::AllocResult;
use crate::erts::process::alloc::*;
use crate::erts::term::prelude::*;
use crate::erts::*;
use crate::mem::bit_size_of;

/// This type represents the old generation process heap
///
/// This heap has no stack, and is only swept when new values are tenured
pub struct OldHeap {
    start: *mut Term,
    end: *mut Term,
    top: *mut Term,
    vheap: VirtualBinaryHeap,
}
impl OldHeap {
    /// Returns a new instance which manages the memory represented
    /// by `start -> start + size`. If `start` is the null pointer,
    /// then this is considered an empty, inactive heap, and will
    /// return sane values for all functions, but will not participate
    /// in collections
    #[inline]
    pub fn new(start: *mut Term, size: usize) -> Self {
        if start.is_null() {
            Self::empty()
        } else {
            let end = unsafe { start.add(size) };
            let top = start;
            let vheap = VirtualBinaryHeap::new(size);
            Self {
                start,
                end,
                top,
                vheap,
            }
        }
    }

    /// Returns an empty, inactive default instance which can be
    /// activated by passing `reset` the same arguments as `new`
    #[inline]
    pub fn empty() -> Self {
        Self {
            start: ptr::null_mut(),
            end: ptr::null_mut(),
            top: ptr::null_mut(),
            vheap: VirtualBinaryHeap::new(0),
        }
    }

    /// Returns true if this heap has been allocated memory,
    /// otherwise returns false. Being inactive implies that
    /// the owning process has not yet undergone tenuring of
    /// objects, or it just completed a full sweep
    #[inline]
    pub fn active(&self) -> bool {
        !self.start.is_null()
    }
}
impl Heap for OldHeap {
    #[inline(always)]
    fn heap_start(&self) -> *mut Term {
        self.start
    }

    #[inline(always)]
    fn heap_top(&self) -> *mut Term {
        self.top
    }

    #[inline(always)]
    fn heap_end(&self) -> *mut Term {
        self.end
    }
}
impl HeapAlloc for OldHeap {
    #[inline]
    unsafe fn alloc_layout(&mut self, layout: Layout) -> AllocResult<NonNull<Term>> {
        let layout = layout.pad_to_align().unwrap();

        let needed = layout.size();
        let available = self.heap_available() * mem::size_of::<Term>();
        if needed >= available {
            return Err(alloc!());
        }

        let top = self.top as *mut u8;
        let new_top = top.add(needed);
        debug_assert!(new_top <= self.end as *mut u8);
        self.top = new_top as *mut Term;

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
impl VirtualHeap<ProcBin> for OldHeap {
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
impl VirtualAllocator<ProcBin> for OldHeap {
    #[inline]
    fn virtual_alloc(&mut self, value: Boxed<ProcBin>) {
        self.vheap.virtual_alloc(value)
    }

    #[inline]
    fn virtual_free(&mut self, value: Boxed<ProcBin>) {
        self.vheap.virtual_free(value)
    }

    #[inline]
    fn virtual_unlink(&mut self, value: Boxed<ProcBin>) {
        self.vheap.virtual_unlink(value)
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
        self.vheap.virtual_clear()
    }
}
impl Default for OldHeap {
    fn default() -> Self {
        Self::empty()
    }
}
impl Drop for OldHeap {
    fn drop(&mut self) {
        unsafe {
            if self.active() {
                // Free virtual binary heap, we can't free the memory of this heap until we've done
                // this
                self.virtual_clear();
                // Free memory region managed by this heap instance
                process::alloc::free(self.heap_start(), self.heap_size());
            }
        }
    }
}
impl fmt::Debug for OldHeap {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_fmt(format_args!(
            "OldHeap (heap: {:p}-{:p}, used: {}, unused: {}) [\n",
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
