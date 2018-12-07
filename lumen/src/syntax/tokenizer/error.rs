use trackable::error::TrackableError;
use trackable::error::{ErrorKind as TrackableErrorKind, ErrorKindExt};
use trackable::*;

/// This crate specific error type.
#[derive(Debug, Clone, TrackableError)]
#[trackable(error_kind = "ErrorKind")]
pub struct Error(TrackableError<ErrorKind>);
impl From<std::num::ParseIntError> for Error {
    fn from(f: std::num::ParseIntError) -> Self {
        ErrorKind::InvalidInput.cause(f).into()
    }
}
impl From<std::num::ParseFloatError> for Error {
    fn from(f: std::num::ParseFloatError) -> Self {
        ErrorKind::InvalidInput.cause(f).into()
    }
}
impl From<num::bigint::ParseBigIntError> for Error {
    fn from(f: num::bigint::ParseBigIntError) -> Self {
        ErrorKind::InvalidInput.cause(f).into()
    }
}

/// The list of the possible error kinds
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    /// Input text is invalid.
    InvalidInput,

    /// Unexpected End-Of-String.
    UnexpectedEos,
}

impl TrackableErrorKind for ErrorKind {}
