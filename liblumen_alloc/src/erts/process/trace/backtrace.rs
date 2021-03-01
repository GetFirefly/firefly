use std::fmt;
use std::iter::FusedIterator;
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

pub type Frame = backtrace::Frame;

use super::{format, utils, Symbolication, TraceFrame};

pub struct Trace {
    frames: ThreadLocalCell<Vec<TraceFrame>>,
    fragment: ThreadLocalCell<Option<NonNull<HeapFragment>>>,
    term: ThreadLocalCell<Option<Term>>,
    top: ThreadLocalCell<Option<Term>>,
}
impl Trace {
    const MAX_FRAMES: usize = 10;

    #[inline]
    fn new() -> Arc<Self> {
        Arc::new(Self {
            frames: ThreadLocalCell::new(Vec::with_capacity(Self::MAX_FRAMES)),
            fragment: ThreadLocalCell::new(None),
            term: ThreadLocalCell::new(None),
            top: ThreadLocalCell::new(None),
        })
    }

    pub fn capture() -> Arc<Self> {
        // Allocates a new trace on the heap
        let trace_arc = Self::new();
        let ptr = Arc::as_ptr(&trace_arc) as *mut Trace;
        let trace = unsafe { &mut *ptr };
        //let stackmap = StackMap::get();

        // Capture the raw metadata for each frame in the trace
        let mut depth = 0;
        backtrace::trace(|frame| {
            // Look up the symbol in our stack map, if we have an
            // entry, then this frame is an Erlang frame, so push
            // it on the trace
            //
            // All other frames can be ignored for now
            //let symbol_address = frame.symbol_address();
            //if stackmap.find_function(symbol_address).is_some() {
            depth += 1;
            trace.push_frame(frame);
            //}

            depth < Self::MAX_FRAMES
        });

        trace_arc
    }

    /// Used by `erlang:raise/3` when the caller can specify a constrained format of `Term` for
    /// the `term` in this `Trace`.
    pub fn from_term(term: Term) -> Arc<Self> {
        let (fragment_term, fragment) = term.clone_to_fragment().unwrap();

        Arc::new(Self {
            frames: Default::default(),
            fragment: ThreadLocalCell::new(Some(fragment)),
            term: ThreadLocalCell::new(Some(fragment_term)),
            top: Default::default(),
        })
    }

    /// Returns the set of native frames in the stack trace
    #[inline]
    pub fn frames(&self) -> &[TraceFrame] {
        self.frames.as_slice()
    }

    #[inline]
    pub fn iter_symbols(&self) -> SymbolIter<'_> {
        SymbolIter::new(self.frames.as_slice(), self.top.as_ref().clone())
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

    /// Sets the top frame of the stacktrace to a specific module/function/arity,
    /// using the provided argument list in place of arity. This is a special case
    /// added to support `undef` or `badarg` errors, which may display the arguments
    /// used to call the function which raised the error.
    ///
    /// NOTE: Support for this is optional, we are not required to display the arguments,
    /// but may do so if available. At a minimum, we do need to support the ability to
    /// add a frame to the trace for calls which we know will fail (e.g. apply/3 with
    /// a function that doesn't exist). This still feels like a gross hack though.
    #[inline]
    pub fn set_top_frame(&self, mfa: &ModuleFunctionArity, arguments: &[Term]) {
        assert!(self.top.is_none(), "top of trace was already set");

        // Get heap to allocate the frame on
        let sizeof_args: usize = arguments
            .iter()
            .map(|t| t.size_in_words() * mem::size_of::<Term>())
            .sum();
        let extra = utils::BASE_FRAME_SIZE + sizeof_args;
        let heap_ptr = self.get_or_create_fragment(extra).unwrap_or(None);
        if let Some(mut heap) = heap_ptr {
            let heap_mut = unsafe { heap.as_mut() };
            if let Ok(frame) = utils::format_mfa(heap_mut, mfa, Some(arguments), None, None) {
                unsafe {
                    self.top.set(Some(frame));
                }
            }
        }
    }

    #[inline]
    pub fn push_frame(&mut self, frame: &Frame) {
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
}

pub struct SymbolIter<'a> {
    frames: &'a [TraceFrame],
    top: Option<Term>,
    pos: Option<usize>,
}
impl<'a> SymbolIter<'a> {
    fn new(frames: &'a [TraceFrame], top: Option<Term>) -> Self {
        Self {
            frames,
            top,
            pos: None,
        }
    }
}

impl Iterator for SymbolIter<'_> {
    type Item = Symbolication;

    fn next(&mut self) -> Option<Self::Item> {
        use std::convert::TryInto;

        if self.pos.is_none() {
            self.pos = Some(0);
        }

        let mut pos = self.pos.unwrap();
        if pos == 0 {
            if let Some(top) = self.top {
                if let Ok(symbol) = top.try_into() {
                    self.pos = Some(1);
                    return Some(symbol);
                }
            }

            // Skip the top
            self.pos = Some(1);
        }

        loop {
            if pos >= self.frames.len() {
                self.pos = Some(pos);
                return None;
            }

            let frame = &self.frames[pos];
            pos += 1;

            if let Some(symbol) = frame.symbolicate() {
                self.pos = Some(pos);
                return Some(symbol.clone());
            }
        }
    }
}
impl FusedIterator for SymbolIter<'_> {}
impl DoubleEndedIterator for SymbolIter<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        use std::convert::TryInto;

        if self.pos.is_none() {
            self.pos = Some(self.frames.len());
        }

        let mut pos = self.pos.unwrap();

        if pos == 0 {
            return None;
        }

        loop {
            if pos == 1 {
                if let Some(top) = self.top {
                    if let Ok(symbol) = top.try_into() {
                        self.pos = Some(0);
                        return Some(symbol);
                    }
                }

                // Skip the top
                self.pos = Some(0);
                return None;
            }

            pos -= 1;
            let frame = &self.frames[pos];
            if let Some(symbol) = frame.symbolicate() {
                self.pos = Some(pos);
                return Some(symbol.clone());
            }
        }
    }
}

// Internals for term construction/allocation
impl Trace {
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
        let mut erlang_frames = if self.top.is_some() {
            Vec::with_capacity(1 + self.frames.len())
        } else {
            Vec::with_capacity(self.frames.len())
        };

        // If top was set, add it as the most recent frame on the stack
        if let Some(top) = self.top.as_ref() {
            erlang_frames.push(*top);
        }

        // Add all of the "real" stack frames
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
    let mut result = None;
    // Otherwise resolve symbols for this frame
    backtrace::resolve_frame(frame, |symbol| {
        let name = symbol.name();
        let mfa = if let Some(name) = name {
            let string = String::from_utf8_lossy(name.as_bytes());
            ModuleFunctionArity::from_symbol_name(string).ok()
        } else {
            None
        };
        let filename = symbol
            .filename()
            .map(|p| p.to_path_buf())
            .or_else(|| symbol.filename_raw().map(|p| p.into_path_buf()));
        let line = symbol.lineno();

        result = Some(Symbolication {
            mfa,
            filename,
            line,
        });
    });
    result
}
