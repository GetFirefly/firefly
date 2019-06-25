use core::cmp;
use core::fmt;

use crate::borrow::CloneToProcess;
use crate::erts::{Node, AllocInProcess};

use super::{AsTerm, Term};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Pid(usize);
impl Pid {
    /// Given a the raw pid value (as a usize), reifies it into a `Pid`
    #[inline]
    pub unsafe fn from_raw(pid: usize) -> Self {
        Self(pid)
    }
}

unsafe impl AsTerm for Pid {
    #[inline]
    unsafe fn as_term(&self) -> Term {
        Term::make_pid(self.0)
    }
}
impl PartialEq<ExternalPid> for Pid {
    #[inline]
    fn eq(&self, _other: &ExternalPid) -> bool {
        false
    }
}
impl PartialOrd<ExternalPid> for Pid {
    #[inline]
    fn partial_cmp(&self, other: &ExternalPid) -> Option<cmp::Ordering> {
        self.partial_cmp(&other.pid)
    }
}

pub struct ExternalPid {
    header: Term,
    node: Node,
    next: *mut u8, // off heap header
    pid: Pid,
}
unsafe impl AsTerm for ExternalPid {
    #[inline]
    unsafe fn as_term(&self) -> Term {
        Term::make_boxed(self as *const Self)
    }
}
impl CloneToProcess for ExternalPid {
    #[inline]
    fn clone_to_process<A: AllocInProcess>(&self, _process: &mut A) -> Term {
        unimplemented!()
    }
}
impl PartialEq<ExternalPid> for ExternalPid {
    fn eq(&self, other: &ExternalPid) -> bool {
        self.node == other.node && self.pid == other.pid
    }
}
impl PartialOrd<ExternalPid> for ExternalPid {
    fn partial_cmp(&self, other: &ExternalPid) -> Option<cmp::Ordering> {
        use cmp::Ordering;
        match self.node.partial_cmp(&other.node) {
            Some(Ordering::Equal) => self.pid.partial_cmp(&other.pid),
            result => result,
        }
    }
}
impl fmt::Debug for ExternalPid {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("ExternalPid")
            .field("header", &self.header.as_usize())
            .field("node", &self.node)
            .field("next", &self.next)
            .field("pid", &self.pid)
            .finish()
    }
}