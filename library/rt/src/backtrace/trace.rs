use alloc::alloc::AllocError;
use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::cell::UnsafeCell;
use core::iter::FusedIterator;
use core::ptr::NonNull;

use firefly_alloc::fragment::HeapFragment;

use crate::term::{Cons, OpaqueTerm, Term, TermFragment};

use super::{Frame, Symbolication, TraceFrame};

/// This struct represents a stack trace that was raised from a process
/// either by an exception, or explicit request. It does not depend on any
/// concrete representation of frames, but instead builds on the `Frame` trait
/// to provide access to details needed to symbolicate and format traces.
pub struct Trace {
    frames: Vec<TraceFrame>,
    fragment: UnsafeCell<Option<TermFragment>>,
}
impl Trace {
    pub const MAX_FRAMES: usize = 10;

    #[inline]
    pub fn new(frames: Vec<TraceFrame>) -> Arc<Self> {
        Arc::new(Self {
            frames,
            fragment: UnsafeCell::new(None),
        })
    }

    #[inline]
    pub fn new_with_term(frames: Vec<TraceFrame>, term: Term) -> Arc<Self> {
        let fragment = TermFragment::new(term).unwrap();
        Arc::new(Self {
            frames,
            fragment: UnsafeCell::new(Some(fragment)),
        })
    }

    #[cfg(feature = "std")]
    pub fn capture() -> Arc<Self> {
        // Allocates a new trace on the heap
        let mut trace_arc = Self::new(Vec::with_capacity(Self::MAX_FRAMES));
        let trace = unsafe { Arc::get_mut_unchecked(&mut trace_arc) };
        //let stackmap = StackMap::get();

        // Capture the raw metadata for each frame in the trace
        let mut depth = 0;
        backtrace::trace(|frame| {
            // The first two frames will always be for backtrace::trace, and Trace::capture,
            // so drop those straight away and do not count them towards our max
            if depth < 2 {
                depth += 1;
                return true;
            }

            // Look up the symbol in our stack map, if we have an
            // entry, then this frame is an Erlang frame, so push
            // it on the trace
            //
            // All other frames can be ignored for now
            //let symbol_address = frame.symbol_address();
            //if stackmap.find_function(symbol_address).is_some() {

            depth += 1;
            trace.push_frame(Box::new(frame.clone()));
            //}

            depth < (Self::MAX_FRAMES + 2)
        });

        trace_arc
    }

    #[cfg(not(feature = "std"))]
    pub fn capture() -> Arc<Self> {
        Self::new(Vec::new())
    }

    /// Returns the set of native frames in the stack trace
    #[inline]
    pub fn frames(&self) -> &[TraceFrame] {
        self.frames.as_slice()
    }

    #[inline]
    pub fn iter_symbols(&self) -> SymbolIter<'_> {
        SymbolIter::new(self.frames.as_slice())
    }

    #[inline]
    pub fn push_frame(&mut self, frame: Box<dyn Frame>) {
        self.frames.push(TraceFrame::from(frame));
    }

    pub fn as_term(&self, args: Option<&[OpaqueTerm]>) -> Result<Term, AllocError> {
        if let Some(fragment) = self.fragment() {
            Ok(fragment.term.into())
        } else {
            self.construct(args)
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

    #[inline]
    fn fragment(&self) -> Option<&TermFragment> {
        unsafe { &*self.fragment.get() }.as_ref()
    }
}

pub struct SymbolIter<'a> {
    frames: &'a [TraceFrame],
    pos: Option<usize>,
}
impl<'a> SymbolIter<'a> {
    fn new(frames: &'a [TraceFrame]) -> Self {
        Self { frames, pos: None }
    }
}
impl Iterator for SymbolIter<'_> {
    type Item = Symbolication;

    fn next(&mut self) -> Option<Self::Item> {
        let mut pos = self.pos.unwrap_or(0);

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
        if self.frames.is_empty() {
            return None;
        }

        let mut pos = self.pos.unwrap_or_else(|| self.frames.len() - 1);

        loop {
            if pos == 0 {
                self.pos = Some(pos);
                return None;
            }

            let frame = &self.frames[pos];
            pos -= 1;

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
    /// that some amount of extra bytes is requested to fulfill auxillary requests.
    fn get_or_create_fragment(
        &self,
        extra: Option<&[OpaqueTerm]>,
    ) -> Result<Option<NonNull<HeapFragment>>, AllocError> {
        if let Some(fragment) = self.fragment() {
            Ok(Some(fragment.fragment.unwrap()))
        } else {
            if let Some(layout) =
                super::symbolication::calculate_fragment_layout(self.frames.len(), extra)
            {
                let heap_ptr = HeapFragment::new(layout, None)?;
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
    fn construct(&self, args: Option<&[OpaqueTerm]>) -> Result<Term, AllocError> {
        assert!(self.fragment().is_none());

        // Either create a heap fragment for the terms, or use the one created already
        let heap_ptr = self.get_or_create_fragment(args)?;
        if heap_ptr.is_none() {
            return Ok(Term::Nil);
        }
        let heap_ptr = heap_ptr.unwrap();
        let heap = unsafe { heap_ptr.as_ref() };

        // If top was set, we have an extra frame to append
        let mut erlang_frames = Vec::with_capacity(self.frames.len());

        // Add all of the "real" stack frames
        for frame in &self.frames[..] {
            if let Some(symbol) = frame.symbolicate() {
                // This implicitly ignores native frames, as symbol.mfa() returns None for those
                if let Some(ref mfa) = symbol.mfa() {
                    let erlang_frame = super::symbolication::format_mfa(
                        mfa,
                        args,
                        symbol.filename(),
                        symbol.line(),
                        heap,
                    )?;
                    erlang_frames.push(erlang_frame.into());
                }
            }
        }

        // Then construct the stacktrace term from the frames we just built up
        let list = Cons::from_slice(erlang_frames.as_slice(), heap)?
            .map(Term::Cons)
            .unwrap_or(Term::Nil);

        // Cache the stacktrace for future queries
        unsafe {
            let fragment = &mut *self.fragment.get();
            *fragment = Some(TermFragment {
                term: list.clone().into(),
                fragment: Some(heap_ptr),
            })
        }

        Ok(list)
    }
}
