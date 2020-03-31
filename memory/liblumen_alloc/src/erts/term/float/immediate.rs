///! This module defines an immediate representation for floating point numbers where
///! the layout consists of a single word containing the f64 value, but encoded via
///! a NaN-boxing approach
///!
///! Where supported, the immediate representation should be preferred.
#[cfg(not(target_arch = "x86_64"))]
compile_error!("Target does not support an immediate float representation");

use core::convert::TryFrom;

use crate::borrow::CloneToProcess;
use crate::erts::exception::AllocResult;
use crate::erts::process::alloc::TermAlloc;
use crate::erts::process::Process;
use crate::erts::term::prelude::{Encode, Term, TypeError, TypedTerm};

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct Float(f64);

impl Float {
    #[inline(always)]
    pub fn new(value: f64) -> Self {
        Self(Self::clamp_value(value))
    }

    #[inline]
    pub fn from_raw(term: *mut Float) -> Self {
        unsafe { *term }
    }

    #[inline(always)]
    pub const fn value(&self) -> f64 {
        self.0
    }

    // NOTE: This is here to provide API parity with packed floats,
    // which are wrapped in `Boxed` in the `TypedTerm` enum
    #[inline(always)]
    pub fn as_ref<'a>(&'a self) -> &'a Self {
        self
    }
}

impl CloneToProcess for Float {
    #[inline]
    fn clone_to_process(&self, _process: &Process) -> Term {
        self.encode().unwrap()
    }

    #[inline]
    fn clone_to_heap<A>(&self, _heap: &mut A) -> AllocResult<Term>
    where
        A: ?Sized + TermAlloc,
    {
        Ok(self.encode().unwrap())
    }

    #[inline(always)]
    fn size_in_words(&self) -> usize {
        1
    }
}

impl TryFrom<TypedTerm> for Float {
    type Error = TypeError;

    fn try_from(typed_term: TypedTerm) -> Result<Self, Self::Error> {
        match typed_term {
            TypedTerm::Float(float) => Ok(float),
            _ => Err(TypeError),
        }
    }
}
