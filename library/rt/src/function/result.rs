use core::convert::Infallible;
use core::fmt;
use core::ops::{self, ControlFlow};

use crate::term::OpaqueTerm;

/// This type reflects the implicit return type expected by the Erlang calling convention
#[derive(Debug, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum ErlangResult<T = OpaqueTerm, E = ()> {
    Ok(T) = 0,
    Err(E) = 1,
    Exit = 2,
}

unsafe impl<T, E> Send for ErlangResult<T, E> {}

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
            Self::Exit => Self::Exit,
        }
    }

    #[inline]
    fn clone_from(&mut self, source: &Self) {
        match (self, source) {
            (Self::Ok(to), Self::Ok(from)) => to.clone_from(from),
            (Self::Err(to), Self::Err(from)) => to.clone_from(from),
            (Self::Exit, Self::Exit) => {}
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
            Self::Exit => panic!("exit cannot be converted to Result<T, E>"),
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
            Self::Err(_) | Self::Exit => None,
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
            Self::Exit => ErlangResult::Exit,
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
            Self::Exit => ErlangResult::Exit,
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
            Self::Exit => unwrap_failed_exit("called `ErlangResult::expect` on an `Exit` value"),
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
            Self::Exit => unwrap_failed_exit("called `ErlangResult::unwrap` on an `Exit` value"),
        }
    }

    #[inline]
    pub fn unwrap_err(self) -> E
    where
        T: fmt::Debug,
    {
        match self {
            Self::Ok(ref v) => {
                unwrap_failed("called `ErlangResult::unwrap_err` on an `Ok` value", v)
            }
            Self::Err(e) => e,
            Self::Exit => {
                unwrap_failed_exit("called `ErlangResult::unwrap_err` on an `Exit` value")
            }
        }
    }

    #[inline]
    pub unsafe fn unwrap_err_unchecked(self) -> E {
        debug_assert!(self.is_err());
        match self {
            // SAFETY: the safety contract must be upheld by the caller.
            Self::Ok(_) | Self::Exit => unsafe { core::hint::unreachable_unchecked() },
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
            Self::Exit => ControlFlow::Break(ErlangResult::Exit),
        }
    }
}
impl<T, E, F: From<E>> ops::FromResidual<ErlangResult<Infallible, E>> for ErlangResult<T, F> {
    #[inline]
    #[track_caller]
    fn from_residual(residual: ErlangResult<Infallible, E>) -> Self {
        match residual {
            ErlangResult::Err(e) => Self::Err(From::from(e)),
            ErlangResult::Exit => Self::Exit,
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

#[inline(never)]
#[cold]
#[track_caller]
fn unwrap_failed_exit(msg: &str) -> ! {
    panic!("{msg}")
}
