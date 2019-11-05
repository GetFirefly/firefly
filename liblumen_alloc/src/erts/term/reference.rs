use core::alloc::Layout;
use core::cmp;
use core::convert::{TryFrom, TryInto};
use core::fmt::{self, Debug, Display};
use core::hash::{Hash, Hasher};
use core::mem::{self, size_of};
use core::ptr;

use alloc::sync::Arc;

use crate::borrow::CloneToProcess;
use crate::erts::exception::system::Alloc;
use crate::erts::term::{arity_of, to_word_size, AsTerm, Boxed, Term, TypeError, TypedTerm};
use crate::erts::{scheduler, HeapAlloc, Node};

pub type Number = u64;

#[derive(Clone, Copy, Eq)]
#[repr(C)]
pub struct Reference {
    #[allow(dead_code)]
    header: Term,
    scheduler_id: scheduler::ID,
    number: Number,
}

impl Reference {
    pub fn need_in_words() -> usize {
        to_word_size(size_of::<Self>())
    }

    /// Create a new `Reference` struct
    pub fn new(scheduler_id: scheduler::ID, number: Number) -> Self {
        Self {
            header: Term::make_header(arity_of::<Self>(), Term::FLAG_REFERENCE),
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

    pub fn number(&self) -> Number {
        self.number
    }
}

unsafe impl AsTerm for Reference {
    #[inline]
    unsafe fn as_term(&self) -> Term {
        Term::make_boxed(self)
    }
}
impl CloneToProcess for Reference {
    fn clone_to_heap<A: HeapAlloc>(&self, heap: &mut A) -> Result<Term, Alloc> {
        unsafe {
            let word_size = self.size_in_words();
            let ptr = heap.alloc(word_size)?.as_ptr() as *mut Self;
            let byte_size = mem::size_of_val(self);
            ptr::copy_nonoverlapping(self as *const Self, ptr, byte_size);

            Ok(Term::make_boxed(ptr))
        }
    }
}
impl Debug for Reference {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Reference")
            .field("header", &format_args!("{:#b}", &self.header.as_usize()))
            .field("scheduler_id", &self.scheduler_id)
            .field("number", &self.number)
            .finish()
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
    fn cmp(&self, other: &Reference) -> cmp::Ordering {
        self.scheduler_id
            .cmp(&other.scheduler_id)
            .then_with(|| self.number.cmp(&other.number))
    }
}
impl PartialEq<Reference> for Reference {
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
impl PartialOrd<Reference> for Reference {
    fn partial_cmp(&self, other: &Reference) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl TryFrom<Term> for Boxed<Reference> {
    type Error = TypeError;

    fn try_from(term: Term) -> Result<Self, Self::Error> {
        term.to_typed_term().unwrap().try_into()
    }
}

impl TryFrom<TypedTerm> for Boxed<Reference> {
    type Error = TypeError;

    fn try_from(typed_term: TypedTerm) -> Result<Self, Self::Error> {
        match typed_term {
            TypedTerm::Boxed(boxed) => match boxed.to_typed_term().unwrap() {
                TypedTerm::Reference(reference) => Ok(reference),
                _ => Err(TypeError),
            },
            _ => Err(TypeError),
        }
    }
}

#[derive(Clone)]
#[repr(C)]
pub struct ExternalReference {
    header: Term,
    arc_node: Arc<Node>,
    reference: Reference,
}

unsafe impl AsTerm for ExternalReference {
    #[inline]
    unsafe fn as_term(&self) -> Term {
        Term::make_boxed(self)
    }
}

impl CloneToProcess for ExternalReference {
    #[inline]
    fn clone_to_heap<A: HeapAlloc>(&self, _heap: &mut A) -> Result<Term, Alloc> {
        unimplemented!()
    }
}

impl Debug for ExternalReference {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("ExternalReference")
            .field("header", &format_args!("{:#b}", &self.header.as_usize()))
            .field("arc_node", &self.arc_node)
            .field("reference", &self.reference)
            .finish()
    }
}

impl Display for ExternalReference {
    fn fmt(&self, _f: &mut fmt::Formatter) -> fmt::Result {
        unimplemented!()
    }
}

impl Hash for ExternalReference {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.arc_node.hash(state);
        self.reference.hash(state);
    }
}

impl PartialEq<Reference> for ExternalReference {
    fn eq(&self, _other: &Reference) -> bool {
        false
    }
}

impl PartialEq<Reference> for Boxed<ExternalReference> {
    fn eq(&self, other: &Reference) -> bool {
        self.as_ref().eq(other)
    }
}

impl PartialEq<Boxed<Reference>> for Boxed<ExternalReference> {
    fn eq(&self, other: &Boxed<Reference>) -> bool {
        self.as_ref().eq(other.as_ref())
    }
}

impl PartialEq<ExternalReference> for ExternalReference {
    fn eq(&self, other: &ExternalReference) -> bool {
        self.arc_node == other.arc_node && self.reference == other.reference
    }
}

impl PartialOrd<Reference> for ExternalReference {
    #[inline]
    fn partial_cmp(&self, _other: &Reference) -> Option<cmp::Ordering> {
        Some(cmp::Ordering::Greater)
    }
}

impl PartialOrd<Reference> for Boxed<ExternalReference> {
    fn partial_cmp(&self, other: &Reference) -> Option<cmp::Ordering> {
        self.as_ref().partial_cmp(other)
    }
}

impl PartialOrd<Boxed<Reference>> for Boxed<ExternalReference> {
    fn partial_cmp(&self, other: &Boxed<Reference>) -> Option<cmp::Ordering> {
        self.as_ref().partial_cmp(other.as_ref())
    }
}

impl PartialOrd<ExternalReference> for ExternalReference {
    fn partial_cmp(&self, other: &ExternalReference) -> Option<cmp::Ordering> {
        use cmp::Ordering;
        match self.arc_node.partial_cmp(&other.arc_node) {
            Some(Ordering::Equal) => self.reference.partial_cmp(&other.reference),
            result => result,
        }
    }
}
