///! This module defines a packed representation for floating point numbers where
///! the layout consists of a header word followed by a raw f64 value.
///!
///! Where supported, the immediate representation should be preferred.
#[cfg(not(target_arch = "x86_64"))]
compile_error!("Target does not support an immediate float representation");

use core::convert::TryFrom;

use crate::borrow::CloneToProcess;
use crate::erts::HeapAlloc;
use crate::erts::process::Process;
use crate::erts::exception::system::Alloc;
use crate::erts::term::prelude::{Term, TypedTerm, TypeError, Encode};


#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct Float(f64);

impl Float {
    #[inline(always)]
    pub const fn new(value: f64) -> Self {
        Self(value)
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
    fn clone_to_heap<A>(&self, _heap: &mut A) -> Result<Term, Alloc>
    where
        A: ?Sized + HeapAlloc,
    {
        Ok(self.encode().unwrap())
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
