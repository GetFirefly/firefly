use core::alloc::Layout;
use core::cmp;
use core::convert::TryFrom;
use core::fmt::{self, Debug, Display};
use core::hash::{Hash, Hasher};

use crate::borrow::CloneToProcess;
use crate::erts::exception::AllocResult;
use crate::erts::node::Node;
use crate::erts::process::alloc::TermAlloc;

use super::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Port(usize);
impl Port {
    /// Given a the raw pid value (as a usize), reifies it into a `Port`
    #[inline]
    pub unsafe fn from_raw(port: usize) -> Self {
        Self(port)
    }

    #[inline(always)]
    pub fn as_usize(self) -> usize {
        self.0
    }
}

impl Display for Port {
    fn fmt(&self, _f: &mut fmt::Formatter) -> fmt::Result {
        unimplemented!()
    }
}

impl PartialEq<ExternalPort> for Port {
    #[inline(always)]
    fn eq(&self, _other: &ExternalPort) -> bool {
        false
    }
}
impl<T> PartialEq<Boxed<T>> for Port
where
    T: PartialEq<Port>,
{
    #[inline]
    default fn eq(&self, other: &Boxed<T>) -> bool {
        other.as_ref().eq(self)
    }
}
impl PartialEq<Boxed<ExternalPort>> for Port {
    #[inline(always)]
    fn eq(&self, _other: &Boxed<ExternalPort>) -> bool {
        false
    }
}
impl PartialOrd<ExternalPort> for Port {
    #[inline]
    fn partial_cmp(&self, other: &ExternalPort) -> Option<cmp::Ordering> {
        self.partial_cmp(&other.port)
    }
}
impl<T> PartialOrd<Boxed<T>> for Port
where
    T: PartialOrd<Port>,
{
    #[inline]
    fn partial_cmp(&self, other: &Boxed<T>) -> Option<cmp::Ordering> {
        other.as_ref().partial_cmp(self).map(|o| o.reverse())
    }
}
impl TryFrom<TypedTerm> for Port {
    type Error = TypeError;

    fn try_from(term: TypedTerm) -> Result<Self, Self::Error> {
        match term {
            TypedTerm::Port(p) => Ok(p),
            _ => Err(TypeError),
        }
    }
}

#[derive(Debug)]
#[repr(C)]
pub struct ExternalPort {
    header: Header<ExternalPort>,
    node: Node,
    next: *mut u8,
    port: Port,
}
impl_static_header!(ExternalPort, Term::HEADER_EXTERN_PORT);
impl CloneToProcess for ExternalPort {
    fn clone_to_heap<A>(&self, _heap: &mut A) -> AllocResult<Term>
    where
        A: ?Sized + TermAlloc,
    {
        unimplemented!()
    }

    fn size_in_words(&self) -> usize {
        crate::erts::to_word_size(Layout::for_value(self).size())
    }
}

impl Display for ExternalPort {
    fn fmt(&self, _f: &mut fmt::Formatter) -> fmt::Result {
        unimplemented!()
    }
}

impl Hash for ExternalPort {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.node.hash(state);
        self.port.hash(state);
    }
}

impl PartialEq for ExternalPort {
    #[inline]
    fn eq(&self, other: &ExternalPort) -> bool {
        self.node == other.node && self.port == other.port
    }
}
impl<T> PartialEq<Boxed<T>> for ExternalPort
where
    T: PartialEq<ExternalPort>,
{
    #[inline]
    fn eq(&self, other: &Boxed<T>) -> bool {
        other.as_ref().eq(self)
    }
}

impl PartialOrd for ExternalPort {
    #[inline]
    fn partial_cmp(&self, other: &ExternalPort) -> Option<cmp::Ordering> {
        use cmp::Ordering;
        match self.node.partial_cmp(&other.node) {
            Some(Ordering::Equal) => self.port.partial_cmp(&other.port),
            result => result,
        }
    }
}
impl<T> PartialOrd<Boxed<T>> for ExternalPort
where
    T: PartialOrd<ExternalPort>,
{
    #[inline]
    fn partial_cmp(&self, other: &Boxed<T>) -> Option<cmp::Ordering> {
        other.as_ref().partial_cmp(self).map(|o| o.reverse())
    }
}

impl TryFrom<TypedTerm> for Boxed<ExternalPort> {
    type Error = TypeError;

    fn try_from(term: TypedTerm) -> Result<Self, Self::Error> {
        match term {
            TypedTerm::ExternalPort(p) => Ok(p),
            _ => Err(TypeError),
        }
    }
}
