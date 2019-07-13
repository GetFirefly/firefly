use core::alloc::AllocErr;
use core::ptr::NonNull;

use crate::erts::term::{ProcBin, Term};

use super::alloc::{self, HeapAlloc, StackAlloc, StackPrimitives, VirtualAlloc};
use super::gc::*;
use super::ProcessControlBlock;

#[derive(Debug)]
#[repr(C)]
pub struct ProcessHeap {
    // the number of minor collections
    pub(super) gen_gc_count: usize,
    // young generation heap
    pub(super) young: YoungHeap,
    // old generation heap
    pub(super) old: OldHeap,
}
impl ProcessHeap {
    pub fn new(heap: *mut Term, heap_size: usize) -> Self {
        let young = YoungHeap::new(heap, heap_size);
        let old = OldHeap::default();
        Self {
            gen_gc_count: 0,
            young,
            old,
        }
    }

    #[inline]
    pub fn should_collect(&self, gc_threshold: f64) -> bool {
        // Check if young generation requires collection
        let used = self.young.heap_used();
        let unused = self.young.unused();
        let threshold = ((used + unused) as f64 * gc_threshold).ceil() as usize;
        if used >= threshold {
            return true;
        }
        // Check if virtual heap size indicates we should do a collection
        let used = self.young.virtual_heap_used();
        let unused = self.young.virtual_heap_unused();
        if unused > 0 {
            let threshold = ((used + unused) as f64 * gc_threshold).ceil() as usize;
            used >= threshold
        } else {
            // We've exceeded the virtual heap size
            true
        }
    }

    #[inline]
    pub fn garbage_collect(
        &mut self,
        process: &ProcessControlBlock,
        need: usize,
        mut rootset: RootSet,
    ) -> Result<usize, GcError> {
        // The primary source of roots we add is the process stack
        rootset.push_range(self.young.stack_pointer(), self.young.stack_size());
        // Initialize the collector
        let mut gc = GarbageCollector::new(self, process, rootset);
        // Run the collector
        gc.collect(need)
    }
}
impl Drop for ProcessHeap {
    fn drop(&mut self) {
        // Free young heap
        let young_heap_start = self.young.heap_start();
        let young_heap_size = self.young.size();
        unsafe { alloc::free(young_heap_start, young_heap_size) };
        // Free old heap, if active
        if self.old.active() {
            let old_heap_start = self.old.heap_start();
            let old_heap_size = self.old.size();
            unsafe { alloc::free(old_heap_start, old_heap_size) };
        }
    }
}
impl HeapAlloc for ProcessHeap {
    #[inline]
    unsafe fn alloc(&mut self, need: usize) -> Result<NonNull<Term>, AllocErr> {
        self.young.alloc(need)
    }

    #[inline]
    fn is_owner<T>(&mut self, ptr: *const T) -> bool {
        if self.young.contains(ptr) || self.old.contains(ptr) {
            return true;
        }
        if self.young.virtual_heap_contains(ptr) || self.old.virtual_heap_contains(ptr) {
            return true;
        }
        false
    }
}
impl VirtualAlloc for ProcessHeap {
    #[inline]
    fn virtual_alloc(&mut self, bin: &ProcBin) -> Term {
        self.young.virtual_alloc(bin)
    }
}
impl StackAlloc for ProcessHeap {
    #[inline]
    unsafe fn alloca(&mut self, need: usize) -> Result<NonNull<Term>, AllocErr> {
        self.young.alloca(need)
    }

    #[inline]
    unsafe fn alloca_unchecked(&mut self, need: usize) -> NonNull<Term> {
        self.young.alloca_unchecked(need)
    }
}
impl StackPrimitives for ProcessHeap {
    #[inline]
    fn stack_size(&self) -> usize {
        self.young.stack_size()
    }

    #[inline]
    unsafe fn set_stack_size(&mut self, size: usize) {
        self.young.set_stack_size(size);
    }

    #[inline]
    fn stack_pointer(&mut self) -> *mut Term {
        self.young.stack_pointer()
    }

    #[inline]
    unsafe fn set_stack_pointer(&mut self, sp: *mut Term) {
        self.young.set_stack_pointer(sp);
    }

    #[inline]
    fn stack_used(&self) -> usize {
        self.young.stack_used()
    }

    #[inline]
    fn stack_available(&self) -> usize {
        self.young.stack_available()
    }

    #[inline]
    fn stack_slot(&mut self, n: usize) -> Option<Term> {
        self.young.stack_slot(n)
    }

    #[inline]
    fn stack_popn(&mut self, n: usize) {
        self.young.stack_popn(n);
    }
}
