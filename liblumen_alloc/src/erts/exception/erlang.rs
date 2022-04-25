use std::alloc::Layout;
use std::mem;
use std::ptr::{self, NonNull};
use std::sync::Arc;

use crate::borrow::CloneToProcess;
use crate::erts::fragment::HeapFragment;
use crate::erts::process::alloc::TermAlloc;
use crate::erts::process::trace::Trace;
use crate::erts::term::prelude::*;

use super::AllocResult;

/// The raw representation of an Erlang panic.
///
/// Initially, this type is allocated on the global heap and not
/// attached to any given process. It can propagate through
/// the exception handling system within a process freely. Once caught and
/// it is being converted to an Erlang term, or if it is being reified by
/// a process somewhere via a trace constructor, the following occurs:
///
/// - The exception reason is cloned to the process heap first
/// - The trace term is constructed from the raw trace
/// - This structure is then bitwise copied to the process heap, but with the fragment set to None
///   and the trace set to a null pointer
/// - The trace field of the copy is then overwritten with the box which points to the trace term.
///
/// When the original exception is cleaned up, this structure is dropped,
/// which deallocates the memory it used on the global heap, and also
/// drops the underlying heap fragment (if there was one). If this was
/// the last outstanding reference to the Trace, then that will also be
/// deallocated.
///
/// Since the layout of this type perfectly matches that of a tuple
/// of the same arity, this makes it a lot more efficient to pass around,
/// as we only need to effectively make a single memcpy and one store to convert it
/// to a tuple on the process heap; this also avoids copying parts of the
/// exception to the process heap that can be allocated globally and shared
/// as the exception propagates; only explicitly modifying the trace term would
/// incur a copy to the process heap. This is especially important as we don't
/// know if we'll be able to allocate much, if any, space on the process heap
/// and our exceptions should not be reliant on doing so except when running
/// the process code.
#[allow(improper_ctypes)]
#[derive(Debug)]
#[repr(C)]
pub struct ErlangException {
    _header: Header<Tuple>,
    kind: Term,
    reason: Term,
    trace: *mut Trace,
    fragment: Option<NonNull<HeapFragment>>,
}
impl ErlangException {
    pub fn new(kind: Term, reason: Term, trace: Arc<Trace>) -> Box<ErlangException> {
        let header = Header::<Tuple>::from_arity(3);
        let trace = Trace::into_raw(trace);
        match reason.size_in_words() {
            0 => Box::new(Self {
                _header: header,
                kind,
                reason,
                trace,
                fragment: None,
            }),
            n => {
                if let Ok(mut nn) = HeapFragment::new_from_word_size(n) {
                    let fragment = unsafe { nn.as_mut() };
                    let reason = reason.clone_to_heap(fragment).unwrap();
                    Box::new(Self {
                        _header: header,
                        kind,
                        reason,
                        trace,
                        fragment: Some(nn),
                    })
                } else {
                    // Fallback to 'unavailable' as reason
                    Box::new(Self {
                        _header: header,
                        kind,
                        reason: Atom::str_to_term("unavailable"),
                        trace,
                        fragment: None,
                    })
                }
            }
        }
    }

    /// When a throw propagates to the top of a process stack, it
    /// gets converted to an exit with `{nocatch, Reason}` as the
    /// exit reason. To facilitate that, we allocate the needed
    /// space in advance if we already required a fragment, otherwise
    /// we create one on demand if possible. If we can't allocate one,
    /// then we have to panic
    pub fn set_nocatch(&mut self) {
        use std::intrinsics::unlikely;

        if unlikely(self.fragment.is_none()) {
            let layout = Tuple::layout_for_len(2);
            let nn = HeapFragment::new(layout).expect("out of memory");
            self.fragment = Some(nn);
        }

        let frag = self
            .fragment
            .map(|nn| unsafe { &mut *nn.as_ptr() })
            .unwrap();
        let mut nocatch_reason = frag.mut_tuple(2).unwrap();

        let nocatch = Atom::str_to_term("nocatch");
        nocatch_reason.set_element(0, nocatch).unwrap();
        nocatch_reason.set_element(1, self.reason).unwrap();

        // Update kind and reason
        self.kind = Atom::EXIT.as_term();
        self.reason = nocatch_reason.into();
    }

    #[inline]
    pub fn kind(&self) -> Term {
        self.kind
    }

    #[inline]
    pub fn reason(&self) -> Term {
        self.reason
    }

    #[inline]
    pub fn fragment(&self) -> Option<NonNull<HeapFragment>> {
        self.fragment
    }

    #[inline]
    pub fn trace(&self) -> Arc<Trace> {
        // HACK(pauls): Manufacture a new reference to the underlying
        // trace. Since we created the exception using Trace::into_raw,
        // we must drop this structure with a corresponding from_raw; if
        // we don't do this little trick here, we would end up dropping
        // the trace prematurely while the exception struct is still live
        let trace = unsafe { Trace::from_raw(self.trace) };
        let result = trace.clone();
        let _ = Trace::into_raw(trace);
        result
    }
}
impl Drop for ErlangException {
    fn drop(&mut self) {
        if let Some(fragment) = self.fragment {
            unsafe {
                fragment.as_ptr().drop_in_place();
            }
        }
        // Drop our trace reference
        let _ = unsafe { Trace::from_raw(self.trace) };
    }
}
impl CloneToProcess for ErlangException {
    fn clone_to_heap<A>(&self, heap: &mut A) -> AllocResult<Term>
    where
        A: ?Sized + TermAlloc,
    {
        let reason = self.reason.clone_to_heap(heap)?;
        let trace = self.trace().as_term()?;
        let tuple = unsafe {
            let ptr = heap
                .alloc_layout(Layout::new::<Self>())?
                .cast::<Self>()
                .as_ptr();

            ptr.write(Self {
                _header: self._header,
                kind: self.kind,
                reason,
                trace: ptr::null_mut(),
                fragment: None,
            });

            let mut tuple = Tuple::from_raw_term(ptr as *mut Term);
            tuple.set_element(2, trace).unwrap();

            tuple
        };

        Ok(tuple.into())
    }

    fn layout(&self) -> Layout {
        // Account for possibility that this could be promoted to a nocatch
        // error, in which case we need a 2-element tuple to fill
        let (layout, _) = Tuple::layout_for_len(3)
            .extend(Tuple::layout_for_len(2))
            .unwrap();
        let tuple_size = layout.size();
        let reason_size = self.reason.size_in_words() * mem::size_of::<Term>();
        let size = tuple_size + reason_size;
        let align = layout.align();

        Layout::from_size_align(size, align).unwrap()
    }

    /// Returns the size in words needed to allocate this value
    #[inline]
    fn size_in_words(&self) -> usize {
        crate::erts::to_word_size(self.layout().size())
    }
}
