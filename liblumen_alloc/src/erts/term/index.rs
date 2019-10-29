use core::convert::{TryFrom, TryInto};
use core::cmp;
use core::ops;
use core::slice;

use thiserror::Error;

use super::prelude::*;

/// A marker trait for index types
pub trait TupleIndex: Into<usize> {}
/// A marker trait for internal index types to help in specialization
pub trait NonPrimitiveIndex: Sized {}

/// This error type is produced when an index is invalid, either due
/// to type or range
#[derive(Error, Debug, Clone, Copy, PartialEq)]
pub enum IndexError {
    #[error("invalid index: bad argument")]
    BadArgument,
    #[error("invalid index {index}, exceeds max length of {len}")]
    OutOfBounds { len: usize, index: usize },
}
impl IndexError {
    pub fn new(index: usize, len: usize) -> Self {
        Self::OutOfBounds { len, index }
    }
}
impl From<core::convert::Infallible> for IndexError {
    fn from(_: core::convert::Infallible) -> Self {
        unreachable!()
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
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
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
impl TupleIndex for OneBasedIndex {}
impl NonPrimitiveIndex for OneBasedIndex {}
impl Default for OneBasedIndex {
    fn default() -> Self {
        Self(1)
    }
}
impl TryFrom<&BigInteger> for OneBasedIndex {
    type Error = IndexError;

    fn try_from(n: &BigInteger) -> Result<Self, Self::Error> {
        Self::new(n.try_into()?)
    }
}
impl TryFrom<Boxed<BigInteger>> for OneBasedIndex {
    type Error = IndexError;

    fn try_from(n: Boxed<BigInteger>) -> Result<Self, Self::Error> {
        Self::new(n.try_into()?)
    }
}
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
            TypedTerm::SmallInteger(n) => n.try_into(),
            TypedTerm::BigInteger(n) => Self::new(n.try_into()?),
            _ => Err(IndexError::BadArgument),
        }
    }
}
impl Into<usize> for OneBasedIndex {
    #[inline]
    fn into(self) -> usize {
        self.0 - 1
    }
}
impl PartialEq<usize> for OneBasedIndex {
    #[inline]
    fn eq(&self, other: &usize) -> bool {
        self.0 == *other
    }
}
impl PartialEq<ZeroBasedIndex> for OneBasedIndex {
    #[inline]
    fn eq(&self, other: &ZeroBasedIndex) -> bool {
        other.eq(self)
    }
}
impl PartialOrd<usize> for OneBasedIndex {
    #[inline]
    fn partial_cmp(&self, other: &usize) -> Option<cmp::Ordering> {
        self.0.partial_cmp(other)
    }
}
impl PartialOrd<ZeroBasedIndex> for OneBasedIndex {
    #[inline]
    fn partial_cmp(&self, other: &ZeroBasedIndex) -> Option<cmp::Ordering> {
        other.partial_cmp(self).map(|o| o.reverse())
    }
}

impl ops::Add for OneBasedIndex {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}
impl ops::Add<usize> for OneBasedIndex {
    type Output = Self;

    fn add(self, rhs: usize) -> Self::Output {
        Self(self.0 + rhs)
    }
}

/// Represents indices which start at 0 and progress upwards
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct ZeroBasedIndex(usize);
impl ZeroBasedIndex {
    #[inline]
    pub fn new(i: usize) -> Self {
        Self(i)
    }
}
impl TupleIndex for ZeroBasedIndex {}
impl NonPrimitiveIndex for ZeroBasedIndex {}
impl Default for ZeroBasedIndex {
    fn default() -> Self {
        Self(0)
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
            TypedTerm::SmallInteger(n) => Ok(Self::new(n.try_into()?)),
            TypedTerm::BigInteger(n) => Ok(Self::new(n.try_into()?)),
            _ => Err(IndexError::BadArgument)
        }
    }
}
impl From<OneBasedIndex> for ZeroBasedIndex {
    #[inline]
    fn from(i: OneBasedIndex) -> ZeroBasedIndex {
        Self(i.0 - 1)
    }
}
impl From<&OneBasedIndex> for ZeroBasedIndex {
    #[inline]
    fn from(i: &OneBasedIndex) -> ZeroBasedIndex {
        Self(i.0 - 1)
    }
}
impl From<usize> for ZeroBasedIndex {
    #[inline]
    fn from(n: usize) -> Self {
        Self::new(n)
    }
}
impl Into<usize> for ZeroBasedIndex {
    #[inline]
    fn into(self) -> usize {
        self.0
    }
}
impl PartialEq<usize> for ZeroBasedIndex {
    #[inline]
    fn eq(&self, other: &usize) -> bool {
        self.0 == *other
    }
}
impl PartialEq<OneBasedIndex> for ZeroBasedIndex {
    fn eq(&self, other: &OneBasedIndex) -> bool {
        let index: ZeroBasedIndex = other.into();
        self.eq(&index)
    }
}
impl PartialOrd<usize> for ZeroBasedIndex {
    #[inline]
    fn partial_cmp(&self, other: &usize) -> Option<cmp::Ordering> {
        self.0.partial_cmp(other)
    }
}
impl PartialOrd<OneBasedIndex> for ZeroBasedIndex {
    fn partial_cmp(&self, other: &OneBasedIndex) -> Option<cmp::Ordering> {
        let index: ZeroBasedIndex = other.into();
        self.partial_cmp(&index)
    }
}

impl ops::Add for ZeroBasedIndex {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}
impl ops::Add<usize> for ZeroBasedIndex {
    type Output = Self;

    fn add(self, rhs: usize) -> Self::Output {
        Self(self.0 + rhs)
    }
}

// Tuple indexing

impl ops::Index<ops::RangeFull> for Tuple {
    type Output = [Term];

    #[inline]
    fn index(&self, index: ops::RangeFull) -> &Self::Output {
        ops::Index::index(self.elements(), index)
    }
}

// Specialization for indexing with usize values (assumed to be zero-based indices)
impl TupleIndex for usize {}

impl ops::Index<usize> for Tuple {
    type Output = Term;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        <usize as slice::SliceIndex<[Term]>>::index(index, self.elements())
    }
}
impl ops::IndexMut<usize> for Tuple {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        <usize as slice::SliceIndex<[Term]>>::index_mut(index, self.elements_mut())
    }
}
impl ops::Index<ops::RangeTo<usize>> for Tuple {
    type Output = [Term];

    #[inline]
    fn index(&self, index: ops::RangeTo<usize>) -> &Self::Output {
        <ops::RangeTo<usize> as slice::SliceIndex<[Term]>>::index(index, self.elements())
    }
}
impl ops::Index<ops::RangeFrom<usize>> for Tuple {
    type Output = [Term];

    #[inline]
    fn index(&self, index: ops::RangeFrom<usize>) -> &Self::Output {
        <ops::RangeFrom<usize> as slice::SliceIndex<[Term]>>::index(index, self.elements())
    }
}

// Generic tuple indexing for any type that implements TupleIndex + NonPrimitiveIndex

impl<I> ops::Index<I> for Tuple
where
    I: TupleIndex + NonPrimitiveIndex,
{
    type Output = Term;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        let uindex: usize = index.into();
        ops::Index::index(self.elements(), uindex)
    }
}

impl<I> ops::IndexMut<I> for Tuple
where
    I: TupleIndex + NonPrimitiveIndex,
{
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        let uindex: usize = index.into();
        ops::IndexMut::index_mut(self.elements_mut(), uindex)
    }
}

impl<I> ops::Index<ops::RangeTo<I>> for Tuple
where
    I: TupleIndex + NonPrimitiveIndex,
{
    type Output = [Term];

    #[inline]
    fn index(&self, index: ops::RangeTo<I>) -> &Self::Output {
        let uindex: usize = index.end.into();
        ops::Index::index(self.elements(), ops::RangeTo { end: uindex })
    }
}


impl<I> ops::Index<ops::RangeFrom<I>> for Tuple
where
    I: TupleIndex + NonPrimitiveIndex,
{
    type Output = [Term];

    #[inline]
    fn index(&self, index: ops::RangeFrom<I>) -> &Self::Output {
        let uindex: usize = index.start.into();
        ops::Index::index(self.elements(), ops::RangeFrom { start: uindex })
    }
}
