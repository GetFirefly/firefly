use alloc::sync::Arc;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::ops::Deref;

use firefly_alloc::clone::WriteCloneIntoRaw;
use firefly_alloc::heap::Heap;

use crate::gc::Gc;
use crate::process::{ProcessId, ProcessIdError};
use crate::services::distribution::Node;

use super::{Boxable, Header, Tag};

#[cfg(not(target_family = "wasm"))]
#[thread_local]
pub static mut CURRENT_PID: Option<Pid> = None;

/// This struct abstracts over the locality of a process identifier
#[repr(C)]
#[derive(Debug, Clone)]
pub struct Pid {
    header: Header,
    id: ProcessId,
    node: Option<Arc<Node>>,
}
impl Boxable for Pid {
    type Metadata = ();

    const TAG: Tag = Tag::Pid;

    #[inline]
    fn header(&self) -> &Header {
        &self.header
    }

    #[inline]
    fn header_mut(&mut self) -> &mut Header {
        &mut self.header
    }

    fn unsafe_clone_to_heap<H: ?Sized + Heap>(&self, heap: &H) -> Gc<Self> {
        let ptr = self as *const Self;
        if heap.contains(ptr.cast()) {
            unsafe { Gc::from_raw(ptr.cast_mut()) }
        } else {
            let mut cloned = Gc::new_uninit_in(heap).unwrap();
            unsafe {
                self.write_clone_into_raw(cloned.as_mut_ptr());
                cloned.assume_init()
            }
        }
    }
}
impl Pid {
    /// Returns the [`Pid`] of the process executing on the current thread
    ///
    /// This function will return `None` if called from a non-scheduler thread, or from a scheduler
    /// thread when no process is currently executing on that thread.
    #[cfg(not(target_family = "wasm"))]
    pub fn current() -> Option<Self> {
        unsafe { CURRENT_PID.clone() }
    }

    #[cfg(target_family = "wasm")]
    pub fn current() -> Option<Self> {
        None
    }

    /// Creates a new local pid, manually.
    ///
    /// This function will return an error if the number/serial components are out of range.
    pub fn new(number: usize, serial: usize) -> Result<Self, ProcessIdError> {
        let id = ProcessId::new(number as u32, serial as u32)?;
        Ok(Self {
            header: Header::new(Tag::Pid, 0),
            id,
            node: None,
        })
    }

    #[inline]
    pub const fn new_local(id: ProcessId) -> Self {
        Self {
            header: Header::new(Tag::Pid, 0),
            id,
            node: None,
        }
    }

    /// Creates a new external pid, manually.
    ///
    /// This function will return an error if the number/serial components are out of range.
    pub fn new_external(
        node: Arc<Node>,
        number: usize,
        serial: usize,
    ) -> Result<Self, ProcessIdError> {
        let id = ProcessId::new(number as u32, serial as u32)?;
        Ok(Self {
            header: Header::new(Tag::Pid, 0),
            id,
            node: Some(node),
        })
    }

    /// Allocates a new local pid, using the global counter.
    ///
    /// NOTE: The pid returned by this function is not guaranteed to be unique. Once the pid
    /// space has been exhausted at least once, pids may be reused, and it is up to the caller
    /// to ensure that a pid is only used by a single live process on the local node at any given
    /// time.
    #[inline]
    pub fn next() -> Self {
        Self {
            header: Header::new(Tag::Pid, 0),
            id: ProcessId::next(),
            node: None,
        }
    }

    /// Returns the raw process identifier
    #[inline]
    pub fn id(&self) -> ProcessId {
        self.id
    }

    /// Returns the node associated with this pid, if applicable
    pub fn node(&self) -> Option<Arc<Node>> {
        self.node.clone()
    }

    #[inline]
    pub fn is_local(&self) -> bool {
        self.node.is_none()
    }

    #[inline]
    pub fn is_external(&self) -> bool {
        self.node.is_some()
    }
}
impl fmt::Display for Pid {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.node {
            None => write!(f, "<0.{}.{}>", self.id.number(), self.id.serial()),
            Some(ref node) => write!(
                f,
                "<{}.{}.{}>",
                node.id(),
                self.id.number(),
                self.id.serial()
            ),
        }
    }
}
impl crate::cmp::ExactEq for Pid {}
impl Eq for Pid {}
impl PartialEq for Pid {
    fn eq(&self, other: &Self) -> bool {
        self.id.eq(&other.id) && self.node.eq(&other.node)
    }
}
impl PartialEq<Gc<Pid>> for Pid {
    fn eq(&self, other: &Gc<Pid>) -> bool {
        self.eq(other.deref())
    }
}
impl Hash for Pid {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
        self.node.hash(state);
    }
}
impl Ord for Pid {
    #[inline]
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        use core::cmp::Ordering;

        match (self.node.as_ref(), other.node.as_ref()) {
            (None, None) => self.id.cmp(&other.id),
            (None, _) => Ordering::Less,
            (Some(a), Some(b)) => a.cmp(b).then(self.id.cmp(&other.id)),
            (Some(_), None) => Ordering::Greater,
        }
    }
}
impl PartialOrd for Pid {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
