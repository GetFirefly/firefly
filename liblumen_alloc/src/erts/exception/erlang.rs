use std::alloc::Layout;
use std::mem;
use std::sync::Arc;

use crate::borrow::CloneToProcess;
use crate::erts::fragment::HeapFragment;
use crate::erts::process::alloc::TermAlloc;
use crate::erts::process::trace::Trace;
use crate::erts::term::prelude::*;

use super::AllocResult;

/// The raw representation of an Erlang panic.
///
/// Initially, this type is allocated in a HeapFragment not
/// attached to any given process. It can propagate through
/// the exception handling system within a process freely. Once caught and
/// is being converted to an Erlang term, or if it is being reified by
/// a process somewhere, the following occurs:
///
/// - The structure is cloned to the process heap
/// - The header is written to make this a 3-element tuple
/// - The "raw" trace pointer is overwritten with the boxed
/// pointer that points to the reified Erlang term form of
/// the trace on the target process heap.
///
/// When the original exception is cleaned up, the heap fragment
/// that was originally allocated is freed.
///
/// Since the layout of this type perfectly matches that of a tuple
/// of the same arity, this makes it a lot more efficient to pass around,
/// as we only need make a single memcpy and one store to convert it
/// to a tuple on the process heap; versus going through all of the
/// runtime machinery to construct a new tuple, especially since we don't
/// know if we'll be able to allocate
#[allow(improper_ctypes)]
#[derive(Debug)]
#[repr(C)]
pub struct ErlangException {
    _header: Header<Tuple>,
    kind: Term,
    reason: Term,
    trace: *mut Trace,
    fragment: Option<*mut HeapFragment>,
}

#[export_name = "__lumen_builtin_raise/2"]
pub extern "C" fn capture_and_raise(kind: Term, reason: Term) -> *mut ErlangException {
    let err = ErlangException::new(kind, reason, Trace::capture());
    Box::into_raw(err)
}

#[export_name = "__lumen_builtin_raise/3"]
pub extern "C" fn raise(kind: Term, reason: Term, trace: *mut Trace) -> *mut ErlangException {
    let err = ErlangException::new(kind, reason, unsafe { Trace::from_raw(trace) });
    Box::into_raw(err)
}

#[export_name = "__lumen_cleanup_exception"]
pub unsafe extern "C" fn cleanup(ptr: *mut ErlangException) {
    let _ = Box::from_raw(ptr);
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
                        fragment: Some(fragment),
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
            self.fragment = Some(nn.as_ptr());
        }

        let frag = self.fragment.map(|ptr| unsafe { &mut *ptr }).unwrap();
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
    pub fn fragment(&self) -> Option<*mut HeapFragment> {
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
                fragment.drop_in_place();
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

        let mut tuple = heap.mut_tuple(3)?;
        tuple.set_element(0, self.kind).unwrap();
        tuple.set_element(1, reason).unwrap();
        tuple.set_element(2, self.trace().as_term()?).unwrap();

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
