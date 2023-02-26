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
            let opaque = unsafe { term.unsafe_move_to_heap(fragment.as_ref()) };
            Ok(Self {
                term: opaque,
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
        if let Some(fragment_ptr) = self.fragment.take() {
            unsafe {
                let range = fragment_ptr.as_ref().used_range();
                range.reap();
                ptr::drop_in_place(fragment_ptr.as_ptr());
            }
        } else {
            self.term.maybe_decrement_refcount();
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::sync::Arc;

    use crate::term::{BinaryData, ListBuilder};
    use firefly_alloc::heap::FixedSizeHeap;

    use super::*;

    #[test]
    fn term_fragment_integration_test() {
        let fragment = TermFragment::new(Term::Nil).unwrap();
        drop(fragment);

        let heap = FixedSizeHeap::<256>::default();
        let mut builder = ListBuilder::new(&heap);
        let bin1 = BinaryData::from_str("testing");
        builder.push(Term::RcBinary(bin1.clone())).unwrap();
        let bin2 = BinaryData::from_small_str("hello", &heap).unwrap();
        builder.push(Term::HeapBinary(bin2)).unwrap();
        let term = builder.finish().map(Term::Cons).unwrap();
        let fragment = TermFragment::new(term).unwrap();
        assert_eq!(Arc::strong_count(&bin1), 2);
        drop(fragment);
        assert_eq!(Arc::strong_count(&bin1), 1);
    }
}
