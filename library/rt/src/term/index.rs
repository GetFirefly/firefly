use core::cmp;
use core::fmt;
use core::ops;
use core::slice;

use super::{OpaqueTerm, Term, Tuple};

/// A marker trait for index types
pub trait TupleIndex: Into<usize> {}
/// A marker trait for internal index types to help in specialization
pub trait NonPrimitiveIndex: Sized {}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct InvalidTupleIndex;
impl fmt::Display for InvalidTupleIndex {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("invalid tuple index")
    }
}
#[cfg(feature = "std")]
impl std::error::Error for InvalidTupleIndex {}

/// Represents indices which start at 1 and progress upwards
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct OneBasedIndex(usize);
impl OneBasedIndex {
    #[inline]
    pub fn new(i: usize) -> Result<Self, InvalidTupleIndex> {
        if i > 0 {
            Ok(Self(i))
        } else {
            Err(InvalidTupleIndex)
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
impl TryFrom<u32> for OneBasedIndex {
    type Error = InvalidTupleIndex;

    #[inline]
    fn try_from(n: u32) -> Result<Self, Self::Error> {
        Self::new(n as usize)
    }
}
impl TryFrom<i64> for OneBasedIndex {
    type Error = InvalidTupleIndex;

    fn try_from(n: i64) -> Result<Self, Self::Error> {
        Self::new(n.try_into().map_err(|_| InvalidTupleIndex)?)
    }
}
impl TryFrom<OpaqueTerm> for OneBasedIndex {
    type Error = InvalidTupleIndex;

    fn try_from(term: OpaqueTerm) -> Result<Self, Self::Error> {
        if !term.is_integer() {
            return Err(InvalidTupleIndex);
        }
        unsafe { term.as_integer().try_into() }
    }
}
impl TryFrom<Term> for OneBasedIndex {
    type Error = InvalidTupleIndex;

    fn try_from(term: Term) -> Result<Self, Self::Error> {
        match term {
            Term::Int(i) if i > 0 => Ok(Self(i.try_into().map_err(|_| InvalidTupleIndex)?)),
            _ => Err(InvalidTupleIndex),
        }
    }
}
impl Into<usize> for OneBasedIndex {
    #[inline(always)]
    fn into(self) -> usize {
        self.0 - 1
    }
}
impl PartialEq<usize> for OneBasedIndex {
    #[inline]
    fn eq(&self, other: &usize) -> bool {
        (self.0 - 1) == *other
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
        (self.0 - 1).partial_cmp(other)
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

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}
impl ops::Add<usize> for OneBasedIndex {
    type Output = Self;

    #[inline]
    fn add(self, rhs: usize) -> Self::Output {
        Self(self.0 + rhs)
    }
}

/// Represents indices which start at 0 and progress upwards
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct ZeroBasedIndex(usize);
impl ZeroBasedIndex {
    #[inline(always)]
    pub fn new(i: usize) -> Self {
        Self(i)
    }
}
impl TupleIndex for ZeroBasedIndex {}
impl NonPrimitiveIndex for ZeroBasedIndex {}
impl Default for ZeroBasedIndex {
    #[inline(always)]
    fn default() -> Self {
        Self(0)
    }
}
impl From<u32> for ZeroBasedIndex {
    #[inline(always)]
    fn from(n: u32) -> Self {
        Self(n as usize)
    }
}
impl TryFrom<i64> for ZeroBasedIndex {
    type Error = InvalidTupleIndex;

    fn try_from(n: i64) -> Result<Self, Self::Error> {
        Ok(Self(n.try_into().map_err(|_| InvalidTupleIndex)?))
    }
}
impl TryFrom<OpaqueTerm> for ZeroBasedIndex {
    type Error = InvalidTupleIndex;

    fn try_from(term: OpaqueTerm) -> Result<Self, Self::Error> {
        if !term.is_integer() {
            return Err(InvalidTupleIndex);
        }
        unsafe { term.as_integer().try_into() }
    }
}
impl TryFrom<Term> for ZeroBasedIndex {
    type Error = InvalidTupleIndex;

    fn try_from(term: Term) -> Result<Self, Self::Error> {
        match term {
            Term::Int(i) => i.try_into().map_err(|_| InvalidTupleIndex),
            _ => Err(InvalidTupleIndex),
        }
    }
}
impl From<OneBasedIndex> for ZeroBasedIndex {
    #[inline]
    fn from(i: OneBasedIndex) -> ZeroBasedIndex {
        Self(i.0 - 1)
    }
}
impl From<usize> for ZeroBasedIndex {
    #[inline(always)]
    fn from(n: usize) -> Self {
        Self(n)
    }
}
impl Into<usize> for ZeroBasedIndex {
    #[inline(always)]
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
        let index: ZeroBasedIndex = (*other).into();
        self.0 == index.0
    }
}
impl PartialOrd<usize> for ZeroBasedIndex {
    #[inline]
    fn partial_cmp(&self, other: &usize) -> Option<core::cmp::Ordering> {
        self.0.partial_cmp(other)
    }
}
impl PartialOrd<OneBasedIndex> for ZeroBasedIndex {
    fn partial_cmp(&self, other: &OneBasedIndex) -> Option<core::cmp::Ordering> {
        let index: ZeroBasedIndex = (*other).into();
        self.partial_cmp(&index)
    }
}

impl ops::Add for ZeroBasedIndex {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}
impl ops::Add<usize> for ZeroBasedIndex {
    type Output = Self;

    #[inline]
    fn add(self, rhs: usize) -> Self::Output {
        Self(self.0 + rhs)
    }
}

// Tuple indexing

impl ops::Index<ops::RangeFull> for Tuple {
    type Output = [OpaqueTerm];

    #[inline]
    fn index(&self, index: ops::RangeFull) -> &Self::Output {
        ops::Index::index(self.as_slice(), index)
    }
}

// Specialization for indexing with usize values (assumed to be zero-based indices)
impl TupleIndex for usize {}

impl ops::Index<usize> for Tuple {
    type Output = OpaqueTerm;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        <usize as slice::SliceIndex<[OpaqueTerm]>>::index(index, self.as_slice())
    }
}
impl ops::IndexMut<usize> for Tuple {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        <usize as slice::SliceIndex<[OpaqueTerm]>>::index_mut(index, self.as_mut_slice())
    }
}
impl ops::Index<ops::RangeTo<usize>> for Tuple {
    type Output = [OpaqueTerm];

    #[inline]
    fn index(&self, index: ops::RangeTo<usize>) -> &Self::Output {
        <ops::RangeTo<usize> as slice::SliceIndex<[OpaqueTerm]>>::index(index, self.as_slice())
    }
}
impl ops::Index<ops::RangeFrom<usize>> for Tuple {
    type Output = [OpaqueTerm];

    #[inline]
    fn index(&self, index: ops::RangeFrom<usize>) -> &Self::Output {
        <ops::RangeFrom<usize> as slice::SliceIndex<[OpaqueTerm]>>::index(index, self.as_slice())
    }
}

// Generic tuple indexing for any type that implements TupleIndex + NonPrimitiveIndex

impl<I> ops::Index<I> for Tuple
where
    I: TupleIndex + NonPrimitiveIndex,
{
    type Output = OpaqueTerm;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        let uindex: usize = index.into();
        ops::Index::index(self.as_slice(), uindex)
    }
}

impl<I> ops::IndexMut<I> for Tuple
where
    I: TupleIndex + NonPrimitiveIndex,
{
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        let uindex: usize = index.into();
        ops::IndexMut::index_mut(self.as_mut_slice(), uindex)
    }
}

impl<I> ops::Index<ops::RangeTo<I>> for Tuple
where
    I: TupleIndex + NonPrimitiveIndex,
{
    type Output = [OpaqueTerm];

    #[inline]
    fn index(&self, index: ops::RangeTo<I>) -> &Self::Output {
        let uindex: usize = index.end.into();
        ops::Index::index(self.as_slice(), ops::RangeTo { end: uindex })
    }
}

impl<I> ops::Index<ops::RangeFrom<I>> for Tuple
where
    I: TupleIndex + NonPrimitiveIndex,
{
    type Output = [OpaqueTerm];

    #[inline]
    fn index(&self, index: ops::RangeFrom<I>) -> &Self::Output {
        let uindex: usize = index.start.into();
        ops::Index::index(self.as_slice(), ops::RangeFrom { start: uindex })
    }
}
