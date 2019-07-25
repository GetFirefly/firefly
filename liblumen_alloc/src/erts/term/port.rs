use core::alloc::AllocErr;
use core::cmp;
use core::fmt;
use core::hash::{Hash, Hasher};

use crate::borrow::CloneToProcess;
use crate::erts::{HeapAlloc, Node};

use super::{AsTerm, Term};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Port(usize);
impl Port {
    /// Given a the raw pid value (as a usize), reifies it into a `Port`
    #[inline]
    pub unsafe fn from_raw(port: usize) -> Self {
        Self(port)
    }
}

unsafe impl AsTerm for Port {
    #[inline]
    unsafe fn as_term(&self) -> Term {
        Term::make_port(self.0)
    }
}
impl PartialEq<ExternalPort> for Port {
    #[inline]
    fn eq(&self, _other: &ExternalPort) -> bool {
        false
    }
}
impl PartialOrd<ExternalPort> for Port {
    #[inline]
    fn partial_cmp(&self, other: &ExternalPort) -> Option<cmp::Ordering> {
        self.partial_cmp(&other.port)
    }
}

pub struct ExternalPort {
    header: Term,
    node: Node,
    next: *mut u8,
    port: Port,
}
unsafe impl AsTerm for ExternalPort {
    #[inline]
    unsafe fn as_term(&self) -> Term {
        Term::make_boxed(self)
    }
}
impl CloneToProcess for ExternalPort {
    fn clone_to_heap<A: HeapAlloc>(&self, _heap: &mut A) -> Result<Term, AllocErr> {
        unimplemented!()
    }
}
impl Hash for ExternalPort {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.node.hash(state);
        self.port.hash(state);
    }
}
impl PartialEq<ExternalPort> for ExternalPort {
    #[inline]
    fn eq(&self, other: &ExternalPort) -> bool {
        self.node == other.node && self.port == other.port
    }
}
impl PartialOrd<ExternalPort> for ExternalPort {
    #[inline]
    fn partial_cmp(&self, other: &ExternalPort) -> Option<cmp::Ordering> {
        use cmp::Ordering;
        match self.node.partial_cmp(&other.node) {
            Some(Ordering::Equal) => self.port.partial_cmp(&other.port),
            result => result,
        }
    }
}
impl fmt::Debug for ExternalPort {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("ExternalPort")
            .field("header", &format_args!("{:#b}", &self.header.as_usize()))
            .field("node", &self.node)
            .field("next", &self.next)
            .field("port", &self.port)
            .finish()
    }
}
