///! This module defines a packed representation for floating point numbers where
///! the layout consists of a header word followed by a raw f64 value.
///!
///! Where supported, the immediate representation should be preferred.
#[cfg(target_arch = "x86_64")]
compile_error!(
    "Packed floats should not be compiled on x86_64, this architecture uses immediate floats!"
);

use core::alloc::Layout;
use core::cmp;
use core::convert::TryFrom;
use core::ptr;

use crate::borrow::CloneToProcess;
use crate::erts::exception::AllocResult;
use crate::erts::process::alloc::TermAlloc;
use crate::impl_static_header;

use crate::erts::term::prelude::{Boxed, Header, Term, TypeError, TypedTerm};

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Float {
    header: Header<Float>,
    value: f64,
}
impl Float {
    #[inline]
    pub fn new(value: f64) -> Self {
        Self {
            header: Default::default(),
            value: Self::clamp_value(value),
        }
    }

    #[inline(always)]
    pub const fn value(&self) -> f64 {
        self.value
    }
}
impl_static_header!(Float, Term::HEADER_FLOAT);

impl Eq for Float {}
impl PartialEq for Float {
    #[inline]
    fn eq(&self, other: &Float) -> bool {
        self.value == other.value
    }
}
impl PartialOrd for Float {
    #[inline]
    fn partial_cmp(&self, other: &Float) -> Option<cmp::Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

impl CloneToProcess for Float {
    #[inline]
    fn clone_to_heap<A>(&self, heap: &mut A) -> AllocResult<Term>
    where
        A: ?Sized + TermAlloc,
    {
        unsafe {
            let layout = Layout::for_value(self);
            let ptr = heap.alloc_layout(layout)?.as_ptr() as *mut Self;
            ptr::copy_nonoverlapping(self as *const Self, ptr, 1);
            Ok(ptr.into())
        }
    }
}

impl TryFrom<TypedTerm> for Boxed<Float> {
    type Error = TypeError;

    fn try_from(typed_term: TypedTerm) -> Result<Self, Self::Error> {
        match typed_term {
            TypedTerm::Float(float) => Ok(float),
            _ => Err(TypeError),
        }
    }
}

impl TryFrom<TypedTerm> for Float {
    type Error = TypeError;

    fn try_from(typed_term: TypedTerm) -> Result<Self, Self::Error> {
        match typed_term {
            TypedTerm::Float(float) => Ok(unsafe { *float.as_ptr() }),
            _ => Err(TypeError),
        }
    }
}

impl Into<f64> for Boxed<Float> {
    fn into(self) -> f64 {
        self.as_ref().value
    }
}
