///! This module defines an error type which distinguishes between runtime and system exceptions.
///!
///! Errors which are part of the normal execution of an Erlang program are represented by the
///! `RuntimeException` type, while errors which are not recoverable from Erlang code are
/// represented ! by the `SystemException` type.
// Allocation errors
mod alloc;
pub use self::alloc::Alloc;

// A wrapper around anyhow::Error that can be cloned and shared across threads
mod arc;
pub use self::arc::ArcError;

// The concrete implementations of the runtime exception classes
mod classes;
pub use self::classes::{Class, Error, Exit, Throw};

// A location represents file/line/column info about an error
mod location;
pub use self::location::Location;

// These helpers provide convenience constructors for common error types
mod helpers;
pub use self::helpers::*;

mod internal;
pub use self::internal::*;

mod runtime;
pub use self::runtime::RuntimeException;

mod system;
pub use self::system::SystemException;

use core::any::type_name;
use core::convert::Into;
use core::marker::PhantomData;

use thiserror::Error;

use super::term::prelude::*;
use crate::erts::string::InvalidEncodingNameError;

/// A convenience type alias for results which fail with `Exception`
pub type Result<T> = core::result::Result<T, Exception>;

/// A convenience type alias for results from allocating functions
pub type AllocResult<T> = core::result::Result<T, Alloc>;

/// An error type which distinguishes between runtime and system exceptions
#[derive(Error, Debug, Clone, PartialEq)]
pub enum Exception {
    #[error("system error")]
    System(#[from] SystemException),
    #[error("runtime error")]
    Runtime(#[from] RuntimeException),
}

// Allows use with ?
impl From<core::convert::Infallible> for Exception {
    fn from(_: core::convert::Infallible) -> Self {
        unreachable!()
    }
}

// System exception type conversions
impl From<Alloc> for Exception {
    fn from(alloc: Alloc) -> Self {
        Self::System(alloc.into())
    }
}
impl From<TermDecodingError> for Exception {
    fn from(err: TermDecodingError) -> Self {
        Self::System(err.into())
    }
}
impl From<TermEncodingError> for Exception {
    fn from(err: TermEncodingError) -> Self {
        Self::System(err.into())
    }
}

// Runtime exception type conversions
impl From<anyhow::Error> for Exception {
    fn from(err: anyhow::Error) -> Self {
        InternalException::from(ArcError::new(err)).into()
    }
}
impl From<InvalidEncodingNameError> for Exception {
    fn from(err: InvalidEncodingNameError) -> Self {
        InternalException::from(ArcError::from_err(err)).into()
    }
}
impl From<InternalException> for Exception {
    fn from(err: InternalException) -> Self {
        match err {
            InternalException::System(err) => Self::System(err),
            InternalException::Internal(err) => Self::Runtime(badarg!(err)),
        }
    }
}

/// Used to represent errors which occur when expecting a
/// particular exception when converting from a more abstract type
#[derive(Error, Debug, Clone, Copy, PartialEq, Eq)]
#[error("unexpected exception occured; expected error of type {}, got {}", type_name::<T>(), type_name::<U>())]
pub struct UnexpectedExceptionError<T, U>(PhantomData<T>, PhantomData<U>)
where
    T: std::error::Error,
    U: std::error::Error;
impl<T, U> Default for UnexpectedExceptionError<T, U>
where
    T: std::error::Error,
    U: std::error::Error,
{
    fn default() -> Self {
        Self(PhantomData, PhantomData)
    }
}
