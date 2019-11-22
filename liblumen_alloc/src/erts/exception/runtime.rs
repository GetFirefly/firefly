use core::array::TryFromSliceError;
use core::convert::TryFrom;
use core::num::TryFromIntError;

use thiserror::Error;

use crate::erts::term::prelude::*;

use super::location::Location;
use super::{Exception, SystemException, UnexpectedExceptionError};

#[derive(Error, Debug, Clone)]
pub enum RuntimeException {
    #[error("{0}")]
    Throw(#[from] super::Throw),
    #[error("{0}")]
    Error(#[from] super::Error),
    #[error("{0}")]
    Exit(#[from] super::Exit),
    #[error("{0}")]
    Unknown(#[from] super::ArcError),
}
impl Eq for RuntimeException {}
impl PartialEq for RuntimeException {
    fn eq(&self, other: &Self) -> bool {
        use RuntimeException::*;
        match (self, other) {
            (Throw(ref lhs), Throw(ref rhs)) => lhs.eq(rhs),
            (Error(ref lhs), Error(ref rhs)) => lhs.eq(rhs),
            (Exit(ref lhs), Exit(ref rhs)) => lhs.eq(rhs),
            (Unknown(_), Unknown(_)) => true,
            _ => false,
        }
    }
}
impl RuntimeException {
    pub fn class(&self) -> Option<super::Class> {
        match self {
            RuntimeException::Throw(e) => Some(e.class()),
            RuntimeException::Exit(e) => Some(e.class()),
            RuntimeException::Error(e) => Some(e.class()),
            RuntimeException::Unknown(_) => None,
        }
    }

    pub fn reason(&self) -> Option<Term> {
        match self {
            RuntimeException::Throw(e) => Some(e.reason()),
            RuntimeException::Exit(e) => Some(e.reason()),
            RuntimeException::Error(e) => Some(e.reason()),
            RuntimeException::Unknown(_err) => None,
        }
    }

    pub fn location(&self) -> Option<Location> {
        match self {
            RuntimeException::Throw(e) => Some(e.location()),
            RuntimeException::Exit(e) => Some(e.location()),
            RuntimeException::Error(e) => Some(e.location()),
            RuntimeException::Unknown(_err) => None,
        }
    }

    pub fn stacktrace(&self) -> Option<Term> {
        match self {
            RuntimeException::Throw(e) => e.stacktrace(),
            RuntimeException::Exit(e) => e.stacktrace(),
            RuntimeException::Error(e) => e.stacktrace(),
            RuntimeException::Unknown(_err) => None,
        }
    }
}

impl From<TryFromIntError> for RuntimeException {
    fn from(_: TryFromIntError) -> Self {
        super::badarg(location!())
    }
}

impl From<TryIntoIntegerError> for RuntimeException {
    fn from(_: TryIntoIntegerError) -> Self {
        super::badarg(location!())
    }
}

impl From<TryFromSliceError> for RuntimeException {
    fn from(_: TryFromSliceError) -> Self {
        super::badarg(location!())
    }
}

impl From<TypeError> for RuntimeException {
    fn from(_: TypeError) -> Self {
        super::badarg(location!())
    }
}
impl TryFrom<Exception> for RuntimeException {
    type Error = UnexpectedExceptionError<RuntimeException, SystemException>;

    fn try_from(err: Exception) -> Result<Self, <Self as TryFrom<Exception>>::Error> {
        match err {
            Exception::Runtime(e) => Ok(e),
            Exception::System(_) => Err(UnexpectedExceptionError::default()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom;
    use crate::erts::exception::Class;

    mod error {
        use super::*;

        #[test]
        fn without_arguments_stores_none() {
            let reason = atom!("badarg");
            let error = error!(reason);

            assert_eq!(error.reason(), Some(reason));
            assert_eq!(error.class(), Some(Class::Error { arguments: None }));
        }

        #[test]
        fn with_arguments_stores_some() {
            let reason = atom!("badarg");
            let arguments = Term::NIL;
            let error = error!(reason, arguments);

            assert_eq!(error.reason(), Some(reason));
            assert_eq!(
                error.class(),
                Some(Class::Error {
                    arguments: Some(arguments)
                })
            );
        }
    }
}
