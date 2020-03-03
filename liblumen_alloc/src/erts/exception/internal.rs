use thiserror::Error;

use crate::erts::term::pid::InvalidPidError;

use super::{Alloc, ArcError, SystemException};
use crate::erts::term::prelude::{TermDecodingError, TermEncodingError};

/// An error type which can only be used internally in `native` code for the runtime.
#[derive(Debug, Error)]
pub enum InternalException {
    #[error("system error")]
    System(#[from] SystemException),
    #[error("internal error")]
    Internal(#[from] ArcError),
}

impl From<anyhow::Error> for InternalException {
    fn from(err: anyhow::Error) -> Self {
        Self::Internal(err.into())
    }
}
impl From<Alloc> for InternalException {
    fn from(err: Alloc) -> Self {
        Self::System(err.into())
    }
}
impl From<InvalidPidError> for InternalException {
    fn from(err: InvalidPidError) -> Self {
        Self::Internal(err.into())
    }
}
impl From<TermDecodingError> for InternalException {
    fn from(err: TermDecodingError) -> Self {
        Self::Internal(err.into())
    }
}
impl From<TermEncodingError> for InternalException {
    fn from(err: TermEncodingError) -> Self {
        Self::Internal(err.into())
    }
}

/// A convenience type alias for results which fail with `InteralException`
pub type InternalResult<T> = Result<T, InternalException>;
