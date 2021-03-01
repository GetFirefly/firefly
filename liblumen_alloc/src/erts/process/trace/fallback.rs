use std::fmt;
use std::mem;
use std::ptr::NonNull;
use std::sync::Arc;

use liblumen_core::util::thread_local::ThreadLocalCell;

use crate::borrow::CloneToProcess;
use crate::erts::exception::ArcError;
use crate::erts::process::alloc::TermAlloc;
use crate::erts::process::{AllocResult, ModuleFunctionArity, Process};
use crate::erts::term::prelude::*;
use crate::erts::HeapFragment;

use super::{format, utils, Symbolication, TraceFrame};

#[derive(Clone)]
pub struct Frame {
    mfa: ModuleFunctionArity,
    args: Option<Vec<Term>>,
}

pub struct Trace {
    frames: ThreadLocalCell<Vec<TraceFrame>>,
    fragment: ThreadLocalCell<Option<NonNull<HeapFragment>>>,
    term: ThreadLocalCell<Option<Term>>,
}
impl Trace {
    const MAX_FRAMES: usize = 10;

    #[inline]
    fn new() -> Arc<Self> {
        Arc::new(Self {
            frames: ThreadLocalCell::new(Vec::with_capacity(Self::MAX_FRAMES)),
            fragment: ThreadLocalCell::new(None),
            term: ThreadLocalCell::new(None),
        })
    }

    pub fn capture() -> Arc<Self> {
        Self::new()
    }

    /// Used by `erlang:raise/3` when the caller can specify a constrained format of `Term` for
    /// the `term` in this `Trace`.
    pub fn from_term(term: Term) -> Arc<Self> {
        let (fragment_term, fragment) = term.clone_to_fragment().unwrap();

        Arc::new(Self {
            frames: Default::default(),
            fragment: ThreadLocalCell::new(Some(fragment)),
            term: ThreadLocalCell::new(Some(fragment_term)),
        })
    }

    /// Returns the set of native frames in the stack trace
    #[inline]
    pub fn frames(&self) -> &[TraceFrame] {
        self.frames.as_slice()
    }

    #[inline]
    pub fn iter_symbols(&self) -> impl Iterator<Item = Symbolication> + DoubleEndedIterator + '_ {
        self.frames.iter().filter_map(|f| f.symbolicate().cloned())
    }

    #[inline]
    pub fn print(
        &self,
        process: &Process,
        kind: Term,
        reason: Term,
        source: Option<ArcError>,
    ) -> std::io::Result<()> {
        format::print(self, process, kind, reason, source)
    }

    #[inline]
    pub fn format(
        &self,
        f: &mut fmt::Formatter,
        process: Option<&Process>,
        kind: Term,
        reason: Term,
        source: Option<ArcError>,
    ) -> std::io::Result<()> {
        format::format(self, f, process, kind, reason, source)
    }

    #[inline]
    pub fn set_top_frame(&self, mfa: &ModuleFunctionArity, arguments: &[Term]) {
        // Get heap to allocate the frame on
        let sizeof_args: usize = arguments
            .iter()
            .map(|t| t.size_in_words() * mem::size_of::<Term>())
            .sum();
        let extra = utils::BASE_FRAME_SIZE + sizeof_args;
        let heap_ptr = self.get_or_create_fragment(extra).unwrap_or(None);
        let frame = if let Some(mut heap) = heap_ptr {
            let heap_mut = unsafe { heap.as_mut() };
            let mut args = Vec::with_capacity(arguments.len());
            for arg in arguments {
                args.push(arg.clone_to_heap(heap_mut).unwrap());
            }

            Frame {
                mfa: mfa.clone(),
                args: Some(args),
            }
        } else {
            Frame {
                mfa: mfa.clone(),
                args: None,
            }
        };

        unsafe {
            self.frames.as_mut().push(TraceFrame::from(frame));
        }
    }

    #[inline]
    pub fn push_frame(&self, frame: &Frame) {
        unsafe {
            self.frames.as_mut().push(TraceFrame::from(frame));
        }
    }

    pub fn as_term(&self) -> AllocResult<Term> {
        if let Some(term) = self.term.as_ref() {
            Ok(*term)
        } else {
            self.construct()
        }
    }

    #[inline(always)]
    pub fn into_raw(trace: Arc<Trace>) -> *mut Trace {
        Arc::into_raw(trace) as *mut Trace
    }

    #[inline(always)]
    pub unsafe fn from_raw(trace: *mut Trace) -> Arc<Trace> {
        Arc::from_raw(trace)
    }

    /// Retrieves the heap fragment allocated for this trace, or creates it,
    /// returning a mutable reference to that heap.
    ///
    /// The allocated size of the fragment is sufficient to hold all of the frames
    /// of the trace in Erlang Term form. The `extra` parameter is used to indicate
    /// that some amount of extra bytes is requested to fulfill auxillary requests,
    /// such as for `top`.
    fn get_or_create_fragment(&self, extra: usize) -> AllocResult<Option<NonNull<HeapFragment>>> {
        if let Some(fragment) = self.fragment.as_ref() {
            Ok(Some(fragment.clone()))
        } else {
            if let Some(layout) = utils::calculate_fragment_layout(self.frames.len(), extra) {
                let heap_ptr = HeapFragment::new(layout)?;
                unsafe {
                    self.fragment.set(Some(heap_ptr.clone()));
                }
                Ok(Some(heap_ptr))
            } else {
                // No fragment needed, nothing to construct
                Ok(None)
            }
        }
    }

    /// Constructs the stacktrace in its Erlang Term form, caching the result
    ///
    /// NOTE: This function should only ever be called once.
    fn construct(&self) -> AllocResult<Term> {
        assert!(self.term.is_none());

        // Either create a heap fragment for the terms, or use the one created already
        let heap_ptr = self.get_or_create_fragment(/* extra= */ 0)?;
        if heap_ptr.is_none() {
            return Ok(Term::NIL);
        }
        let mut heap_ptr = heap_ptr.unwrap();
        let heap = unsafe { heap_ptr.as_mut() };

        // If top was set, we have an extra frame to append
        let mut erlang_frames = Vec::with_capacity(self.frames.len());

        // Add all of the stack frames
        for frame in &self.frames[..] {
            if let Some(symbol) = frame.symbolicate() {
                if let Some(ref mfa) = symbol.module_function_arity() {
                    let erlang_frame =
                        utils::format_mfa(heap, mfa, None, symbol.filename(), symbol.line())?;
                    erlang_frames.push(erlang_frame);
                }
            }
        }

        // Then construct the stacktrace term from the frames we just built up
        let list = heap.list_from_slice(erlang_frames.as_slice())?;
        let term: Term = list.into();

        // Cache the stacktrace for future queries
        unsafe { self.term.set(Some(term)) };

        Ok(term)
    }
}

pub(super) fn resolve_frame(frame: &Frame) -> Option<Symbolication> {
    Some(Symbolication {
        mfa: Some(frame.mfa),
        ..Default::default()
    })
}
