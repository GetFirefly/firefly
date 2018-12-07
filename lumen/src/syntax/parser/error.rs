use trackable::error::TrackableError;
use trackable::error::{ErrorKind as TrackableErrorKind, ErrorKindExt};
use trackable::*;

use crate::syntax::preprocessor;
use crate::syntax::tokenizer;
use crate::syntax::tokenizer::LexicalToken;

/// This crate specific error type.
#[derive(Debug, Clone, TrackableError)]
#[trackable(error_kind = "ErrorKind")]
pub struct Error(TrackableError<ErrorKind>);
impl From<tokenizer::Error> for Error {
    fn from(f: tokenizer::Error) -> Self {
        match *f.kind() {
            tokenizer::ErrorKind::InvalidInput => ErrorKind::InvalidInput.takes_over(f).into(),
            tokenizer::ErrorKind::UnexpectedEos => ErrorKind::UnexpectedEos.takes_over(f).into(),
        }
    }
}
impl From<preprocessor::Error> for Error {
    fn from(f: preprocessor::Error) -> Self {
        match f.kind().clone() {
            preprocessor::ErrorKind::InvalidInput => ErrorKind::InvalidInput.takes_over(f).into(),
            preprocessor::ErrorKind::UnexpectedToken(t) => {
                ErrorKind::UnexpectedToken(t).takes_over(f).into()
            }
            preprocessor::ErrorKind::UnexpectedEos => ErrorKind::UnexpectedEos.takes_over(f).into(),
        }
    }
}

/// The list of the possible error kinds
#[derive(Debug, Clone)]
pub enum ErrorKind {
    InvalidInput,
    UnexpectedToken(LexicalToken),
    UnexpectedEos,
    Other,
}
impl TrackableErrorKind for ErrorKind {}
