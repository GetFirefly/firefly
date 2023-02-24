use alloc::alloc::AllocError;
use core::ptr::{self, NonNull};

use firefly_alloc::fragment::HeapFragment;

use super::{OpaqueTerm, Term, Value};

/// A term fragment is used for situations in which a single term needs a lifetime
/// separate from that of any process or port, such as those associated with internal
/// runtime structures such as monitors/links.
pub struct TermFragment {
    /// The raw opaque term represented by this fragment
    pub term: OpaqueTerm,
    /// The fragment this term is allocated in, if needed
    pub fragment: Option<NonNull<HeapFragment>>,
}
// TermFragment is designed to be sent between threads, but is not thread safe
unsafe impl Send for TermFragment {}
impl TermFragment {
    /// Moves `term` into a new `TermFragment` using its `clone_to_heap` implementation
    #[inline]
    pub fn new(term: Term) -> Result<Self, AllocError> {
        if term.is_immediate() || term.is_refcounted() {
            Ok(Self {
                term: term.into(),
                fragment: None,
            })
        } else {
            let layout = term.layout();
            let fragment = HeapFragment::new(layout, None)?;
            let term = unsafe { term.unsafe_clone_to_heap(fragment.as_ref()) };
            Ok(Self {
                term: term.into(),
                fragment: Some(fragment),
            })
        }
    }

    /// Clones `source` into a new `TermFragment` using its `clone_to_heap` implementation
    pub fn clone_from(source: &Term) -> Result<Self, AllocError> {
        if source.is_immediate() || source.is_refcounted() {
            Ok(Self {
                term: source.clone().into(),
                fragment: None,
            })
        } else {
            let layout = source.layout();
            let fragment = HeapFragment::new(layout, None)?;
            let term = unsafe { source.unsafe_clone_to_heap(fragment.as_ref()) };
            Ok(Self {
                term: term.into(),
                fragment: Some(fragment),
            })
        }
    }
}
impl Drop for TermFragment {
    fn drop(&mut self) {
        use crate::gc::Reap;
        use firefly_alloc::heap::Heap;

        // If a heap fragment was required, reap it of dead references and drop
        if let Some(fragment) = self.fragment {
            unsafe {
                let range = fragment.as_ref().used_range();
                range.reap();
                ptr::drop_in_place(fragment.as_ptr());
            }
        } else {
            self.term.maybe_decrement_refcount();
        }
    }
}
