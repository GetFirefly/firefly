mod heap;
mod stack;

use alloc::alloc::{AllocError, Allocator, Layout};
use core::cell::UnsafeCell;
use core::ptr::NonNull;

use liblumen_alloc::heap::Heap;

use crate::error::ErlangException;
use crate::function::ModuleFunctionArity;
use crate::term::ProcessId;

pub use self::heap::ProcessHeap;
pub use self::stack::ProcessStack;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ProcessStatus {
    Running,
    Runnable,
    Waiting,
    Exiting,
    Errored(NonNull<ErlangException>),
}

pub struct Process {
    parent: Option<ProcessId>,
    pid: ProcessId,
    #[allow(dead_code)]
    mfa: ModuleFunctionArity,
    /// The process status is only ever manipulated/accessed by the owning scheduler
    status: UnsafeCell<ProcessStatus>,
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
    stack: UnsafeCell<ProcessStack>,
}
impl Process {
    pub fn new(parent: Option<ProcessId>, pid: ProcessId, mfa: ModuleFunctionArity) -> Self {
        Self {
            parent,
            pid,
            mfa,
            status: UnsafeCell::new(ProcessStatus::Waiting),
            heap: UnsafeCell::new(ProcessHeap::new()),
            stack: UnsafeCell::new(ProcessStack::new(32).unwrap()),
        }
    }

    pub fn parent(&self) -> Option<ProcessId> {
        self.parent
    }

    pub fn pid(&self) -> ProcessId {
        self.pid
    }

    pub fn status(&self) -> ProcessStatus {
        unsafe { self.status.get().read() }
    }

    pub fn stack(&self) -> &ProcessStack {
        unsafe { &*self.stack.get() }
    }

    pub fn exit_normal(&self) {
        unsafe {
            self.set_status(ProcessStatus::Exiting);
        }
    }

    pub fn exit_error(&self, exception: NonNull<ErlangException>) {
        unsafe {
            self.set_status(ProcessStatus::Errored(exception));
        }
    }

    /// Sets the process status
    ///
    /// # Safety
    ///
    /// This function must be called with exclusive access to the process status, i.e. by
    /// the owning scheduler. It is not safe to set this status from anywhere else,
    /// except from within the process itself, as while the process is executing it
    /// has exclusive access to its own status.
    pub unsafe fn set_status(&self, status: ProcessStatus) {
        self.status.get().write(status);
    }

    #[inline(always)]
    fn heap(&self) -> &ProcessHeap {
        unsafe { &*self.heap.get() }
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
    fn contains<T: ?Sized>(&self, ptr: *const T) -> bool {
        self.heap().contains(ptr)
    }
}
