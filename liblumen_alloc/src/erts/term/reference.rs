use core::alloc::AllocErr;
use core::cmp;
use core::fmt;
use core::mem;
use core::ptr;

use crate::borrow::CloneToProcess;
use crate::erts::{to_word_size, HeapAlloc, Node};

use super::{AsTerm, Term};

#[cfg(target_pointer_width = "32")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Reference([u32; 3]);

#[cfg(target_pointer_width = "64")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Reference([u64; 2]);

impl Reference {
    /// Reifies a `Reference` from a raw pointer
    pub unsafe fn from_raw(ptr: *mut Reference) -> Self {
        *ptr
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
            let size = self.size_in_words();
            let ptr = heap.alloc(size)?.as_ptr() as *mut Self;
            ptr::copy_nonoverlapping(self as *const Self, ptr, size);
            Ok(Term::make_boxed(ptr))
        }
    }

    fn size_in_words(&self) -> usize {
        to_word_size(mem::size_of_val(self))
    }
}
impl PartialEq<ExternalReference> for Reference {
    #[inline]
    fn eq(&self, _other: &ExternalReference) -> bool {
        false
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
            .field("header", &self.header.as_usize())
            .field("node", &self.node)
            .field("next", &self.next)
            .field("reference", &self.reference)
            .finish()
    }
}
