use alloc::alloc::{AllocError, Layout};
use alloc::boxed::Box;
use core::ptr::NonNull;

use intrusive_collections::intrusive_adapter;
use intrusive_collections::{LinkedList, LinkedListLink};

use firefly_alloc::fragment::HeapFragment;

use crate::services::registry::WeakAddress;
use crate::term::OpaqueTerm;

use super::Priority;

// An intrusive linked list adapter for storing boxed signal entries
intrusive_adapter!(pub SystemTaskAdapter = Box<SystemTask>: SystemTask { link: LinkedListLink });

/// A type alias for the intrusive linked list type
pub type SystemTaskList = LinkedList<SystemTaskAdapter>;

pub const MAX_SYSTEM_TASK_ARGS: usize = 2;

pub struct SystemTask {
    link: LinkedListLink,
    pub ty: SystemTaskType,
    pub requestor: WeakAddress,
    pub priority: Priority,
    pub reply_tag: OpaqueTerm,
    pub request_id: OpaqueTerm,
    pub args: [OpaqueTerm; MAX_SYSTEM_TASK_ARGS],
    fragment: NonNull<HeapFragment>,
}
impl SystemTask {
    /// Allocates a new `SystemTask` with a heap fragment of `layout` size and alignment.
    ///
    /// The resulting `SystemTask` is not valid for placement in a process system task queue,
    /// the caller is expected to finish initializing it with appropriate task metadata.
    pub fn new(ty: SystemTaskType, layout: Layout) -> Result<Box<Self>, AllocError> {
        let fragment = HeapFragment::new(layout, None)?;
        Ok(Box::new(Self {
            link: LinkedListLink::new(),
            ty,
            requestor: WeakAddress::System,
            priority: Priority::default(),
            reply_tag: OpaqueTerm::NONE,
            request_id: OpaqueTerm::NONE,
            args: [OpaqueTerm::NONE; MAX_SYSTEM_TASK_ARGS],
            fragment,
        }))
    }

    #[inline(always)]
    pub fn fragment(&self) -> &HeapFragment {
        unsafe { self.fragment.as_ref() }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum SystemTaskType {
    GcMajor = 1,
    GcMinor,
    /// This is only used to support ERTS tests
    Test,
}
