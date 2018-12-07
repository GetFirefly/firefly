use trackable::error::TrackableError;
use trackable::error::{ErrorKind as TrackableErrorKind, ErrorKindExt};
use trackable::*;

use crate::syntax::tokenizer::{self, LexicalToken};

/// This crate specific error type.
#[derive(Debug, Clone, TrackableError)]
#[trackable(error_kind = "ErrorKind")]
pub struct Error(TrackableError<ErrorKind>);
impl Error {
    pub(crate) fn unexpected_token(token: LexicalToken) -> Self {
        ErrorKind::UnexpectedToken(token).into()
    }
}
impl From<tokenizer::Error> for Error {
    fn from(e: tokenizer::Error) -> Self {
        let kind = match *e.kind() {
            tokenizer::ErrorKind::InvalidInput => ErrorKind::InvalidInput,
            tokenizer::ErrorKind::UnexpectedEos => ErrorKind::UnexpectedEos,
        };
        kind.takes_over(e).into()
    }
}
impl From<std::env::VarError> for Error {
    fn from(f: std::env::VarError) -> Self {
        ErrorKind::InvalidInput.cause(f).into()
    }
}
impl From<std::io::Error> for Error {
    fn from(f: std::io::Error) -> Self {
        ErrorKind::InvalidInput.cause(f).into()
    }
}
impl From<glob::PatternError> for Error {
    fn from(f: glob::PatternError) -> Self {
        ErrorKind::InvalidInput.cause(f).into()
    }
}
impl From<glob::GlobError> for Error {
    fn from(f: glob::GlobError) -> Self {
        ErrorKind::InvalidInput.cause(f).into()
    }
}

/// The list of the possible error kinds
#[derive(Debug, Clone)]
pub enum ErrorKind {
    /// Input is invalid.
    InvalidInput,

    /// Unexpected token.
    UnexpectedToken(LexicalToken),

    /// Unexpected End-Of-String.
    UnexpectedEos,
}
impl TrackableErrorKind for ErrorKind {}
