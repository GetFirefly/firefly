use core::alloc::Layout;
use core::cmp::Ordering;
use core::convert::TryFrom;
use core::fmt::{self, Debug, Display};
use core::hash::{Hash, Hasher};
use core::mem;
use core::ptr;

use crate::borrow::CloneToProcess;
use crate::erts::exception::AllocResult;
use crate::erts::{scheduler, HeapAlloc, Node};

use super::prelude::{Term, TypeError, TypedTerm, Boxed, Header};

pub type ReferenceNumber = u64;

#[derive(Debug, Clone, Copy, Eq)]
#[repr(C)]
pub struct Reference {
    header: Header<Reference>,
    scheduler_id: scheduler::ID,
    number: ReferenceNumber,
}

impl Reference {
    /// Create a new `Reference` struct
    pub fn new(scheduler_id: scheduler::ID, number: ReferenceNumber) -> Self {
        Self {
            header: Default::default(),
            scheduler_id,
            number,
        }
    }

    /// Reifies a `Reference` from a raw pointer
    pub unsafe fn from_raw(ptr: *mut Reference) -> Self {
        *ptr
    }

    /// This function produces a `Layout` which represents the memory layout
    /// needed for the reference header, scheduler ID and number.
    #[inline]
    pub const fn layout() -> Layout {
        let size = mem::size_of::<Self>();
        unsafe { Layout::from_size_align_unchecked(size, mem::align_of::<Term>()) }
    }

    pub fn scheduler_id(&self) -> scheduler::ID {
        self.scheduler_id
    }

    pub fn number(&self) -> ReferenceNumber {
        self.number
    }
}

impl CloneToProcess for Reference {
    fn clone_to_heap<A>(&self, heap: &mut A) -> AllocResult<Term>
    where
        A: ?Sized + HeapAlloc,
    {
        unsafe {
            let layout = Layout::new::<Self>();
            let byte_size = layout.size();
            let ptr = heap.alloc_layout(layout)?.as_ptr() as *mut Self;
            ptr::copy_nonoverlapping(self as *const Self, ptr, byte_size);

            Ok(ptr.into())
        }
    }
}
impl Display for Reference {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "#Reference<0.{}.{}>", self.scheduler_id, self.number)
    }
}
impl Hash for Reference {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.scheduler_id.hash(state);
        self.number.hash(state);
    }
}
impl Ord for Reference {
    fn cmp(&self, other: &Reference) -> Ordering {
        self.scheduler_id
            .cmp(&other.scheduler_id)
            .then_with(|| self.number.cmp(&other.number))
    }
}
impl PartialEq for Reference {
    fn eq(&self, other: &Reference) -> bool {
        (self.scheduler_id == other.scheduler_id) && (self.number == other.number)
    }
}
impl PartialEq<ExternalReference> for Reference {
    #[inline]
    fn eq(&self, _other: &ExternalReference) -> bool {
        false
    }
}
impl<T> PartialEq<Boxed<T>> for Reference
where
    T: PartialEq<Reference>,
{
    #[inline]
    default fn eq(&self, other: &Boxed<T>) -> bool {
        other.as_ref().eq(self)
    }
}
impl PartialEq<Boxed<ExternalReference>> for Reference {
    #[inline(always)]
    fn eq(&self, _other: &Boxed<ExternalReference>) -> bool {
        false
    }
}

impl PartialOrd<Reference> for Reference {
    fn partial_cmp(&self, other: &Reference) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<T> PartialOrd<Boxed<T>> for Reference
where
    T: PartialOrd<Reference>,
{
    fn partial_cmp(&self, other: &Boxed<T>) -> Option<Ordering> {
        other.as_ref().partial_cmp(self).map(|o| o.reverse())
    }
}

impl TryFrom<TypedTerm> for Boxed<Reference> {
    type Error = TypeError;

    fn try_from(typed_term: TypedTerm) -> Result<Self, Self::Error> {
        match typed_term {
            TypedTerm::Reference(reference) => Ok(reference),
            _ => Err(TypeError),
        }
    }
}

#[derive(Debug, Clone)]
#[repr(C)]
pub struct ExternalReference {
    header: Header<ExternalReference>,
    node: Node,
    next: *mut u8,
    reference: Reference,
}

impl CloneToProcess for ExternalReference {
    #[inline]
    fn clone_to_heap<A>(&self, _heap: &mut A) -> AllocResult<Term>
    where
        A: ?Sized + HeapAlloc,
    {
        unimplemented!()
    }
}

impl Display for ExternalReference {
    fn fmt(&self, _f: &mut fmt::Formatter) -> fmt::Result {
        unimplemented!()
    }
}

impl Hash for ExternalReference {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.node.hash(state);
        self.reference.hash(state);
    }
}

impl PartialEq for ExternalReference {
    fn eq(&self, other: &ExternalReference) -> bool {
        self.node == other.node && self.reference == other.reference
    }
}
impl PartialEq<Reference> for ExternalReference {
    fn eq(&self, _other: &Reference) -> bool {
        false
    }
}
impl<T> PartialEq<Boxed<T>> for ExternalReference
where
    T: PartialEq<ExternalReference>,
{
    #[inline]
    default fn eq(&self, other: &Boxed<T>) -> bool {
        other.as_ref().eq(self)
    }
}
impl PartialEq<Boxed<Reference>> for ExternalReference {
    #[inline(always)]
    fn eq(&self, _other: &Boxed<Reference>) -> bool {
        false
    }
}

impl PartialOrd for ExternalReference {
    fn partial_cmp(&self, other: &ExternalReference) -> Option<Ordering> {
        match self.node.partial_cmp(&other.node) {
            Some(Ordering::Equal) => self.reference.partial_cmp(&other.reference),
            result => result,
        }
    }
}
impl PartialOrd<Reference> for ExternalReference {
    #[inline]
    fn partial_cmp(&self, _other: &Reference) -> Option<Ordering> {
        Some(Ordering::Greater)
    }
}
impl<T> PartialOrd<Boxed<T>> for ExternalReference
where
    T: PartialOrd<ExternalReference>,
{
    #[inline]
    fn partial_cmp(&self, other: &Boxed<T>) -> Option<Ordering> {
        other.as_ref().partial_cmp(self).map(|o| o.reverse())
    }
}
