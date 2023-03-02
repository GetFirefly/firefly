use alloc::boxed::Box;

use core::convert::Infallible;
use core::ops::{self, ControlFlow};

use crate::process::Generator;
use crate::term::OpaqueTerm;

use super::ModuleFunctionArity;

/// This type reflects the implicit return type expected by the Erlang calling convention
#[derive(Debug)]
#[repr(u8)]
pub enum ErlangResult {
    /// Function returned successfully with the given value
    Ok(OpaqueTerm) = 0,
    /// Function raised an error during execution, stored in the process state
    Err,
    /// Function caused the process to exit
    Exit,
    /// Function yielded during execution, with the given continuation
    Await(Box<Generator>),
    /// Function is requesting to trampoline to another function, called "trapping" in ERTS
    ///
    /// When this happens, it is expected that the caller
    Trap(&'static ModuleFunctionArity),
}

unsafe impl Send for ErlangResult {}

impl Clone for ErlangResult {
    #[inline]
    fn clone(&self) -> Self {
        match self {
            Self::Ok(x) => Self::Ok(*x),
            Self::Err => Self::Err,
            Self::Exit => Self::Exit,
            Self::Await(gen) => Self::Await(gen.clone()),
            Self::Trap(x) => Self::Trap(*x),
        }
    }
}
impl From<Result<OpaqueTerm, ()>> for ErlangResult {
    fn from(result: Result<OpaqueTerm, ()>) -> Self {
        match result {
            Ok(v) => Self::Ok(v),
            Err(_) => Self::Err,
        }
    }
}
impl ErlangResult {
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
    pub fn is_ready(&self) -> bool {
        match self {
            Self::Await(_) | Self::Trap(_) => false,
            _ => true,
        }
    }

    #[inline]
    pub fn unwrap(self) -> OpaqueTerm {
        match self {
            Self::Ok(v) => v,
            Self::Err => unwrap_failed("called `ErlangResult::unwrap` on an `Err` value"),
            Self::Exit => unwrap_failed("called `ErlangResult::unwrap` on an `Exit` value"),
            Self::Await(_) => unwrap_failed("called `ErlangResult::unwrap` on an `Await` value"),
            Self::Trap(_) => unwrap_failed("called `ErlangResult::unwrap` on a `Trap` value"),
        }
    }
}

impl ops::Try for ErlangResult {
    type Output = OpaqueTerm;
    type Residual = ErlangResult;

    #[inline]
    fn from_output(output: Self::Output) -> Self {
        Self::Ok(output)
    }

    #[inline]
    fn branch(self) -> ControlFlow<Self::Residual, Self::Output> {
        match self {
            Self::Ok(v) => ControlFlow::Continue(v),
            Self::Err => ControlFlow::Break(ErlangResult::Err),
            Self::Exit => ControlFlow::Break(ErlangResult::Exit),
            Self::Await(cont) => ControlFlow::Break(ErlangResult::Await(cont)),
            Self::Trap(cont) => ControlFlow::Break(ErlangResult::Trap(cont)),
        }
    }
}
impl ops::FromResidual<ErlangResult> for ErlangResult {
    #[inline]
    #[track_caller]
    fn from_residual(residual: ErlangResult) -> Self {
        residual
    }
}
impl ops::FromResidual<Result<Infallible, ()>> for ErlangResult {
    #[inline]
    #[track_caller]
    fn from_residual(_residual: Result<Infallible, ()>) -> Self {
        Self::Err
    }
}
impl ops::Residual<OpaqueTerm> for ErlangResult {
    type TryType = ErlangResult;
}

#[inline(never)]
#[cold]
#[track_caller]
fn unwrap_failed(msg: &str) -> ! {
    panic!("{msg}")
}
