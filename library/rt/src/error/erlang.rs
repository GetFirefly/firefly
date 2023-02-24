use alloc::boxed::Box;
use alloc::sync::Arc;
use core::ptr::NonNull;

use firefly_alloc::fragment::HeapFragment;

use crate::backtrace::Trace;
use crate::term::{atoms, Atom, OpaqueTerm, Term};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ExceptionClass {
    Error,
    Exit,
    Throw,
}
impl TryFrom<Atom> for ExceptionClass {
    type Error = ();

    fn try_from(a: Atom) -> Result<Self, ()> {
        if a == atoms::Error {
            Ok(Self::Error)
        } else if a == atoms::Exit {
            Ok(Self::Exit)
        } else if a == atoms::Throw {
            Ok(Self::Throw)
        } else {
            Err(())
        }
    }
}
impl Into<Atom> for ExceptionClass {
    fn into(self) -> Atom {
        match self {
            Self::Error => atoms::Error,
            Self::Exit => atoms::Exit,
            Self::Throw => atoms::Throw,
        }
    }
}
impl Into<OpaqueTerm> for ExceptionClass {
    fn into(self) -> OpaqueTerm {
        let atom: Atom = self.into();
        atom.into()
    }
}
impl Into<Term> for ExceptionClass {
    fn into(self) -> Term {
        Term::Atom(self.into())
    }
}

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
#[derive(Debug)]
#[repr(C)]
pub struct ErlangException {
    pub kind: ExceptionClass,
    pub reason: OpaqueTerm,
    pub meta: OpaqueTerm,
    trace: *mut Trace,
    pub fragment: Option<NonNull<HeapFragment>>,
}
impl ErlangException {
    pub fn new(kind: Atom, reason: Term, trace: Arc<Trace>) -> Box<Self> {
        let trace = Trace::into_raw(trace);

        Box::new(Self {
            kind: kind.try_into().unwrap(),
            reason: reason.into(),
            meta: OpaqueTerm::NIL,
            trace,
            fragment: None,
        })
    }

    pub fn new_with_meta(kind: Atom, reason: Term, meta: Term, trace: Arc<Trace>) -> Box<Self> {
        let trace = Trace::into_raw(trace);

        Box::new(Self {
            kind: kind.try_into().unwrap(),
            reason: reason.into(),
            meta: meta.into(),
            trace,
            fragment: None,
        })
    }

    /// Convert a boxed `ErlangException` into a `NonNull<ErlangException>`
    pub fn into_raw(self: Box<Self>) -> NonNull<Self> {
        unsafe { NonNull::new_unchecked(Box::into_raw(self)) }
    }

    /// Convert a pointer produced by `ErlangException::as_ptr` into a `Box<ErlangException>`
    ///
    /// # SAFETY
    ///
    /// Calls to this function _must_ be paired with a call to `as_ptr`, otherwise you risk a double-free
    pub unsafe fn from_raw(ptr: *mut Self) -> Box<Self> {
        assert!(!ptr.is_null());
        Box::from_raw(ptr)
    }

    #[inline]
    pub fn kind(&self) -> Atom {
        self.kind.into()
    }

    #[inline]
    pub fn reason(&self) -> Term {
        self.reason.into()
    }

    #[inline]
    pub fn meta(&self) -> Term {
        self.meta.into()
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
