mod heap;

use alloc::alloc::{AllocError, Allocator, Layout};
use core::cell::UnsafeCell;
use core::ptr::NonNull;

use liblumen_alloc::heap::Heap;

use crate::term::ProcessId;

use self::heap::ProcessHeap;

pub struct Process {
    parent: Option<ProcessId>,
    pid: ProcessId,
    /// The process heap can be safely accessed directly via UnsafeCell because it
    /// is always the case that either:
    ///
    /// * The heap is being mutated by the process itself
    /// * The process has yielded and the scheduler is doing garbage collection
    ///
    /// In both cases access is exclusive, and special care is taken to guarantee
    /// that when a GC takes place, that live references held by the suspended process
    /// are properly updated so that the aliasing in that case is safe.
    heap: UnsafeCell<ProcessHeap>,
}
impl Process {
    pub fn new(parent: Option<ProcessId>, pid: ProcessId) -> Self {
        Self {
            parent,
            pid,
            heap: UnsafeCell::new(ProcessHeap::new()),
        }
    }

    #[inline(always)]
    fn heap(&self) -> &ProcessHeap {
        &*self.heap.get()
    }
}

unsafe impl Allocator for Process {
    #[inline]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.heap().allocate(layout)
    }

    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        self.heap().deallocate(ptr, layout)
    }

    #[inline]
    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        self.heap().grow(ptr, old_layout, new_layout)
    }

    #[inline]
    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        self.heap().grow_zeroed(ptr, old_layout, new_layout)
    }

    #[inline]
    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        self.heap().shrink(ptr, old_layout, new_layout)
    }
}

impl Heap for Process {
    #[inline]
    fn heap_start(&self) -> *mut u8 {
        self.heap().heap_start()
    }

    #[inline]
    fn heap_top(&self) -> *mut u8 {
        self.heap().heap_top()
    }

    #[inline]
    fn heap_end(&self) -> *mut u8 {
        self.heap().heap_end()
    }

    #[inline]
    fn high_water_mark(&self) -> Option<NonNull<u8>> {
        self.heap().high_water_mark()
    }

    #[inline]
    fn contains(&self, ptr: *const u8) -> bool {
        self.heap().contains()
    }
}
