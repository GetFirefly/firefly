///! This module defines an error type which distinguishes between runtime and system exceptions.
///!
///! Errors which are part of the normal execution of an Erlang program are represented by the
///! `RuntimeException` type, while errors which are not recoverable from Erlang code are represented
///! by the `SystemException` type.

// Allocation errors
mod alloc;
pub use self::alloc::Alloc;

// A wrapper around anyhow::Error that can be cloned and shared across threads
mod arc;
pub use self::arc::ArcError;

// The concrete implementations of the runtime exception classes
mod classes;
pub use self::classes::{Class, Exit, Throw, Error};

// A location represents file/line/column info about an error
mod location;
pub use self::location::Location;

// These helpers provide convenience constructors for common error types
mod helpers;
pub use self::helpers::*;

mod runtime;
pub use self::runtime::RuntimeException;

mod system;
pub use self::system::SystemException;


use core::convert::Into;

use thiserror::Error;

use super::term::prelude::*;
use super::string::InvalidEncodingNameError;

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
impl From<AtomError> for Exception {
    fn from(atom_error: AtomError) -> Self {
        Self::Runtime(atom_error.into())
    }
}
impl From<BoolError> for Exception {
    fn from(bool_error: BoolError) -> Self {
        Self::Runtime(bool_error.into())
    }
}
impl From<BytesFromBinaryError> for Exception {
    fn from(err: BytesFromBinaryError) -> Self {
        use BytesFromBinaryError::*;

        match err {
            NotABinary | Type => Self::Runtime(badarg(location!())),
            Alloc(e) => Self::System(e.into()),
        }
    }
}
impl From<InvalidEncodingNameError> for Exception {
    fn from(encoding_error: InvalidEncodingNameError) -> Self {
        Self::Runtime(encoding_error.into())
    }
}
impl From<ImproperList> for Exception {
    fn from(improper_list: ImproperList) -> Self {
        Self::Runtime(improper_list.into())
    }
}
impl From<IndexError> for Exception {
    fn from(index_error: IndexError) -> Self {
        Self::Runtime(index_error.into())
    }
}
impl From<InvalidPidError> for Exception {
    fn from(_err: InvalidPidError) -> Self {
        Self::Runtime(badarg(location!()))
    }
}
impl From<StrFromBinaryError> for Exception {
    fn from(err: StrFromBinaryError) -> Self {
        use StrFromBinaryError::*;

        match err {
            NotABinary | Type | Utf8Error(_) => Self::Runtime(badarg(location!())),
            Alloc(e) => Self::System(e.into()),
        }
    }
}
impl From<core::num::TryFromIntError> for Exception {
    fn from(try_from_int_error: core::num::TryFromIntError) -> Self {
        Self::Runtime(try_from_int_error.into())
    }
}
impl From<TryIntoIntegerError> for Exception {
    fn from(try_into_integer_error: TryIntoIntegerError) -> Self {
        Self::Runtime(try_into_integer_error.into())
    }
}
impl From<TypeError> for Exception {
    fn from(type_error: TypeError) -> Self {
        Self::Runtime(type_error.into())
    }
}
