use core::cmp;

use crate::borrow::CloneToProcess;
use crate::erts::{Node, ProcessControlBlock};

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
        Term::from_raw(self.0 | Term::FLAG_PORT)
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

#[derive(Debug)]
pub struct ExternalPort {
    header: Term,
    node: Node,
    next: *mut u8,
    port: Port,
}
unsafe impl AsTerm for ExternalPort {
    #[inline]
    unsafe fn as_term(&self) -> Term {
        Term::from_raw(self as *const _ as usize | Term::FLAG_BOXED)
    }
}
impl CloneToProcess for ExternalPort {
    #[inline]
    fn clone_to_process(&self, _process: &mut ProcessControlBlock) -> Term {
        unimplemented!()
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
