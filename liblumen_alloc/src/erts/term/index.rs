use core::convert::{TryFrom, TryInto};
use core::fmt::{self, Display};

use crate::erts::term::{BigInteger, Boxed, SmallInteger, Term, TryIntoIntegerError, TypedTerm};

pub fn try_from_one_based_term_to_zero_based_usize(index_term: Term) -> Result<usize, Error> {
    let index_one_based: OneBased = index_term.try_into()?;
    let index_zero_based: ZeroBased = index_one_based.into();

    Ok(index_zero_based.into())
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Error {
    BadArgument,
    OutOfBounds { len: usize, index: usize },
}

impl Error {
    pub fn new(index: usize, len: usize) -> Self {
        Self::OutOfBounds { len, index }
    }
}

impl Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::OutOfBounds { len, index } => {
                write!(f, "invalid index {}: exceeds max length of {}", index, len)
            }
            Self::BadArgument => write!(f, "invalid index: bad argument"),
        }
    }
}

impl From<TryIntoIntegerError> for Error {
    fn from(_: TryIntoIntegerError) -> Self {
        Self::BadArgument
    }
}

pub struct OneBased(usize);

impl TryFrom<Boxed<BigInteger>> for OneBased {
    type Error = Error;

    fn try_from(boxed_big_integer: Boxed<BigInteger>) -> Result<Self, Self::Error> {
        let u: usize = boxed_big_integer.try_into()?;

        Ok(OneBased(u))
    }
}

impl TryFrom<SmallInteger> for OneBased {
    type Error = Error;

    fn try_from(small_integer: SmallInteger) -> Result<Self, Self::Error> {
        match small_integer.try_into() {
            Ok(u) if 0 < u => Ok(OneBased(u)),
            _ => Err(Error::BadArgument),
        }
    }
}

impl TryFrom<Term> for OneBased {
    type Error = Error;

    fn try_from(term: Term) -> Result<Self, Self::Error> {
        term.to_typed_term().unwrap().try_into()
    }
}

impl TryFrom<TypedTerm> for OneBased {
    type Error = Error;

    fn try_from(typed_term: TypedTerm) -> Result<Self, Self::Error> {
        match typed_term {
            TypedTerm::SmallInteger(small_integer) => small_integer.try_into(),
            TypedTerm::Boxed(boxed) => boxed.to_typed_term().unwrap().try_into(),
            TypedTerm::BigInteger(big_integer) => big_integer.try_into(),
            _ => Err(Error::BadArgument),
        }
    }
}

pub struct ZeroBased(usize);

impl From<OneBased> for ZeroBased {
    fn from(one_based: OneBased) -> ZeroBased {
        ZeroBased(one_based.0 - 1)
    }
}

impl Into<usize> for ZeroBased {
    fn into(self: ZeroBased) -> usize {
        self.0
    }
}
