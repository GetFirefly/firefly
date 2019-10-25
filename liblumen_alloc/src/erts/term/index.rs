use core::convert::{TryFrom, TryInto};
use core::fmt;

use super::prelude::*;

/// This error type is produced when an index is invalid, either due
/// to type or range
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IndexError {
    BadArgument,
    OutOfBounds { len: usize, index: usize },
}
impl IndexError {
    pub fn new(index: usize, len: usize) -> Self {
        Self::OutOfBounds { len, index }
    }
}
impl fmt::Display for IndexError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::OutOfBounds { len, index } => {
                write!(f, "invalid index {}: exceeds max length of {}", index, len)
            }
            Self::BadArgument => write!(f, "invalid index: bad argument"),
        }
    }
}
impl From<TryIntoIntegerError> for IndexError {
    fn from(_: TryIntoIntegerError) -> Self {
        Self::BadArgument
    }
}
impl From<core::num::TryFromIntError> for IndexError {
    fn from(_: core::num::TryFromIntError) -> Self {
        Self::BadArgument
    }
}

/// Represents indices which start at 1 and progress upwards
#[repr(transparent)]
pub struct OneBasedIndex(usize);
impl OneBasedIndex {
    #[inline]
    pub fn new(i: usize) -> Result<Self, IndexError> {
        if i > 0 {
            Ok(Self(i))
        } else {
            Err(IndexError::BadArgument)
        }
    }
}
/*
impl TryFrom<BigInteger> for OneBasedIndex {
    type Error = IndexError;

    fn try_from(n: BigInteger) -> Result<Self, Self::Error> {
        Self::new(n.into())
    }
}
impl TryFrom<Boxed<BigInteger>> for OneBasedIndex {
    type Error = IndexError;

    fn try_from(n: Boxed<BigInteger>) -> Result<Self, Self::Error> {
        Self::new(n.into())
    }
}
*/
impl TryFrom<SmallInteger> for OneBasedIndex {
    type Error = IndexError;

    fn try_from(n: SmallInteger) -> Result<Self, Self::Error> {
        Self::new(n.try_into()?)
    }
}
impl TryFrom<Term> for OneBasedIndex {
    type Error = IndexError;

    fn try_from(term: Term) -> Result<Self, Self::Error> {
        term.decode().unwrap().try_into()
    }
}
impl TryFrom<TypedTerm> for OneBasedIndex {
    type Error = IndexError;

    fn try_from(term: TypedTerm) -> Result<Self, Self::Error> {
        match term {
            TypedTerm::SmallInteger(n) => n.try_into().map_err(|_| IndexError::BadArgument),
            //TypedTerm::BigInteger(n) => Self::new((*n.as_ref()).into()),
            _ => Err(IndexError::BadArgument),
        }
    }
}
impl Into<usize> for OneBasedIndex {
    fn into(self) -> usize {
        self.0
    }
}

/// Represents indices which start at 0 and progress upwards
#[repr(transparent)]
pub struct ZeroBasedIndex(usize);
impl ZeroBasedIndex {
    #[inline]
    pub fn new(i: usize) -> Self {
        Self(i)
    }
}
impl TryFrom<Term> for ZeroBasedIndex {
    type Error = IndexError;

    fn try_from(term: Term) -> Result<Self, Self::Error> {
        term.decode().unwrap().try_into()
    }
}
impl TryFrom<TypedTerm> for ZeroBasedIndex {
    type Error = IndexError;

    fn try_from(term: TypedTerm) -> Result<Self, Self::Error> {
        match term {
            TypedTerm::SmallInteger(n) => Ok(Self::new(n.try_into().map_err(|_| IndexError::BadArgument)?)),
            //TypedTerm::BigInteger(n) => Ok(Self::new(n.try_into().map_err(|_| IndexError::BadArgument)?)),
            _ => Err(IndexError::BadArgument)
        }
    }
}
impl From<OneBasedIndex> for ZeroBasedIndex {
    fn from(i: OneBasedIndex) -> ZeroBasedIndex {
        Self(i.0 - 1)
    }
}
impl Into<usize> for ZeroBasedIndex {
    fn into(self) -> usize {
        self.0
    }
}
