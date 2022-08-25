mod apply;
mod mfa;

pub use self::apply::*;
pub use self::mfa::ModuleFunctionArity;

use core::convert::Infallible;
use core::fmt;
use core::ops::{self, ControlFlow};
use core::ptr::NonNull;

use crate::error::ErlangException;
use crate::term::{Atom, OpaqueTerm};

/// This type reflects the implicit return type expected by the Erlang calling convention
#[derive(Debug, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum ErlangResult<T = OpaqueTerm, E = NonNull<ErlangException>> {
    Ok(T) = 0,
    Err(E) = 1,
}
impl<T, E> const Clone for ErlangResult<T, E>
where
    T: ~const Clone + ~const core::marker::Destruct,
    E: ~const Clone + ~const core::marker::Destruct,
{
    #[inline]
    fn clone(&self) -> Self {
        match self {
            Self::Ok(x) => Self::Ok(x.clone()),
            Self::Err(x) => Self::Err(x.clone()),
        }
    }

    #[inline]
    fn clone_from(&mut self, source: &Self) {
        match (self, source) {
            (Self::Ok(to), Self::Ok(from)) => to.clone_from(from),
            (Self::Err(to), Self::Err(from)) => to.clone_from(from),
            (to, from) => *to = from.clone(),
        }
    }
}
impl<T, E> From<Result<T, E>> for ErlangResult<T, E> {
    fn from(result: Result<T, E>) -> Self {
        match result {
            Ok(v) => Self::Ok(v),
            Err(v) => Self::Err(v),
        }
    }
}
impl<T, E> Into<Result<T, E>> for ErlangResult<T, E> {
    fn into(self) -> Result<T, E> {
        match self {
            Self::Ok(v) => Ok(v),
            Self::Err(v) => Err(v),
        }
    }
}
impl<T, E> ErlangResult<T, E> {
    #[inline]
    pub fn is_ok(&self) -> bool {
        match self {
            Self::Ok(_) => true,
            _ => false,
        }
    }

    #[inline]
    pub fn is_err(&self) -> bool {
        !self.is_ok()
    }

    #[inline]
    pub fn ok(self) -> Option<T> {
        match self {
            Self::Ok(v) => Some(v),
            Self::Err(_) => None,
        }
    }

    #[inline]
    pub fn map<U, F>(self, op: F) -> ErlangResult<U, E>
    where
        F: FnOnce(T) -> U,
    {
        match self {
            Self::Ok(v) => ErlangResult::Ok(op(v)),
            Self::Err(e) => ErlangResult::Err(e),
        }
    }

    #[inline]
    pub fn map_err<F, O>(self, op: O) -> ErlangResult<T, F>
    where
        O: FnOnce(E) -> F,
    {
        match self {
            Self::Ok(v) => ErlangResult::Ok(v),
            Self::Err(e) => ErlangResult::Err(op(e)),
        }
    }

    #[inline]
    pub fn expect(self, msg: &str) -> T
    where
        E: fmt::Debug,
    {
        match self {
            Self::Ok(v) => v,
            Self::Err(ref e) => unwrap_failed(msg, e),
        }
    }

    #[inline]
    pub fn unwrap(self) -> T
    where
        E: fmt::Debug,
    {
        match self {
            Self::Ok(v) => v,
            Self::Err(ref e) => unwrap_failed("called `ErlangResult::unwrap` on an `Err` value", e),
        }
    }

    #[inline]
    pub fn unwrap_err(self) -> E
    where
        T: fmt::Debug,
    {
        match self {
            Self::Ok(ref v) => {
                unwrap_failed("called `ErlangResultt::unwrap_err` on an `Ok` value", v)
            }
            Self::Err(e) => e,
        }
    }

    #[inline]
    pub unsafe fn unwrap_err_unchecked(self) -> E {
        debug_assert!(self.is_err());
        match self {
            // SAFETY: the safety contract must be upheld by the caller.
            Self::Ok(_) => unsafe { core::hint::unreachable_unchecked() },
            Self::Err(e) => e,
        }
    }
}

impl<T, E> ops::Try for ErlangResult<T, E> {
    type Output = T;
    type Residual = ErlangResult<Infallible, E>;

    #[inline]
    fn from_output(output: Self::Output) -> Self {
        Self::Ok(output)
    }

    #[inline]
    fn branch(self) -> ControlFlow<Self::Residual, Self::Output> {
        match self {
            Self::Ok(v) => ControlFlow::Continue(v),
            Self::Err(e) => ControlFlow::Break(ErlangResult::Err(e)),
        }
    }
}
impl<T, E, F: From<E>> ops::FromResidual<ErlangResult<Infallible, E>> for ErlangResult<T, F> {
    #[inline]
    #[track_caller]
    fn from_residual(residual: ErlangResult<Infallible, E>) -> Self {
        match residual {
            ErlangResult::Err(e) => Self::Err(From::from(e)),
            _ => unreachable!(),
        }
    }
}
impl<T, E, F: From<E>> ops::FromResidual<Result<Infallible, E>> for ErlangResult<T, F> {
    #[inline]
    #[track_caller]
    fn from_residual(residual: Result<Infallible, E>) -> Self {
        match residual {
            Err(e) => Self::Err(From::from(e)),
            _ => unreachable!(),
        }
    }
}
impl<T, E> ops::Residual<T> for ErlangResult<Infallible, E> {
    type TryType = ErlangResult<T, E>;
}

#[inline(never)]
#[cold]
#[track_caller]
fn unwrap_failed(msg: &str, error: &dyn fmt::Debug) -> ! {
    panic!("{msg}: {error:?}")
}

/// This struct represents the serialized form of a symbol table entry
///
/// This struct is intentionally laid out in memory to be identical to
/// `ModuleFunctionArity` with an extra field (the function pointer).
/// This allows the symbol table to use ModuleFunctionArity without
/// requiring
///
/// NOTE: This struct must have a size that is a power of 8
#[repr(C, align(8))]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct FunctionSymbol {
    /// Module name atom
    pub module: Atom,
    /// Function name atom
    pub function: Atom,
    /// The arity of the function
    pub arity: u8,
    /// An opaque pointer to the function
    ///
    /// To call the function, it is necessary to transmute this
    /// pointer to one of the correct type. All Erlang functions
    /// expect terms, and return a term as result.
    ///
    /// NOTE: The target type must be marked `extern "C"`, in order
    /// to ensure that the correct calling convention is used.
    pub ptr: *const (),
}

/// Function symbols are read-only and pinned, and therefore Sync
unsafe impl Sync for FunctionSymbol {}

/// Function symbols are read-only and pinned, and therefore Send
unsafe impl Send for FunctionSymbol {}
