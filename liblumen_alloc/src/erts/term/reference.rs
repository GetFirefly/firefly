use core::alloc::{AllocErr, Layout};
use core::cmp;
use core::fmt::{self, Debug};
use core::mem;
use core::ptr;

use crate::borrow::CloneToProcess;
use crate::erts::{scheduler, HeapAlloc, Node};

use super::{AsTerm, Term};

pub type Number = u64;

#[derive(Clone, Copy, Eq)]
pub struct Reference {
    #[allow(dead_code)]
    header: Term,
    scheduler_id: scheduler::ID,
    number: Number,
}

impl Reference {
    /// Create a new `Reference` struct
    pub fn new(scheduler_id: scheduler::ID, number: Number) -> Self {
        Self {
            header: Term::make_header(0, Term::FLAG_REFERENCE),
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
    fn clone_to_heap<A: HeapAlloc>(&self, heap: &mut A) -> Result<Term, AllocErr> {
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
impl PartialOrd<ExternalReference> for Reference {
    #[inline]
    fn partial_cmp(&self, other: &ExternalReference) -> Option<cmp::Ordering> {
        self.partial_cmp(&other.reference)
    }
}

#[derive(Clone)]
#[repr(C)]
pub struct ExternalReference {
    header: Term,
    node: Node,
    next: *mut u8,
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
    fn clone_to_heap<A: HeapAlloc>(&self, _heap: &mut A) -> Result<Term, AllocErr> {
        unimplemented!()
    }
}
impl PartialEq<ExternalReference> for ExternalReference {
    fn eq(&self, other: &ExternalReference) -> bool {
        self.node == other.node && self.reference == other.reference
    }
}
impl PartialOrd<ExternalReference> for ExternalReference {
    fn partial_cmp(&self, other: &ExternalReference) -> Option<cmp::Ordering> {
        use cmp::Ordering;
        match self.node.partial_cmp(&other.node) {
            Some(Ordering::Equal) => self.reference.partial_cmp(&other.reference),
            result => result,
        }
    }
}
impl fmt::Debug for ExternalReference {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("ExternalReference")
            .field("header", &format_args!("{:#b}", &self.header.as_usize()))
            .field("node", &self.node)
            .field("next", &self.next)
            .field("reference", &self.reference)
            .finish()
    }
}
