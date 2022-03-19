use std::num::NonZeroUsize;
use std::ops::{Add, AddAssign, Sub, SubAssign};

use codespan::{ByteIndex, ByteOffset, RawIndex, RawOffset};

use super::SourceId;

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SourceIndex(NonZeroUsize);
impl SourceIndex {
    const INDEX_MASK: usize = u32::max_value() as usize;

    const UNKNOWN_SRC_ID: usize = (SourceId::UNKNOWN_SOURCE_ID as usize) << 32;

    pub const UNKNOWN: Self = Self(unsafe { NonZeroUsize::new_unchecked(Self::UNKNOWN_SRC_ID) });

    #[inline]
    pub fn new(source: SourceId, index: ByteIndex) -> Self {
        let source = (source.get() as usize) << 32;

        Self(NonZeroUsize::new(source | index.0 as usize).unwrap())
    }

    #[inline]
    pub fn source_id(&self) -> SourceId {
        let source_id_part = (self.0.get() >> 32) as u32;
        if source_id_part == SourceId::UNKNOWN_SOURCE_ID {
            SourceId::UNKNOWN
        } else {
            SourceId::new(source_id_part)
        }
    }

    #[inline]
    pub fn index(&self) -> ByteIndex {
        ByteIndex((self.0.get() & Self::INDEX_MASK) as u32)
    }

    pub fn to_usize(&self) -> usize {
        self.0.get()
    }
}
impl Default for SourceIndex {
    fn default() -> Self {
        Self::UNKNOWN
    }
}

impl Add<usize> for SourceIndex {
    type Output = SourceIndex;

    #[inline]
    fn add(self, rhs: usize) -> Self {
        if self == Self::UNKNOWN {
            return Self::UNKNOWN;
        }
        let source = self.source_id();
        let index = self.index();
        let new_index = index.0 as RawOffset + rhs as RawOffset;
        Self::new(source, ByteIndex(new_index as RawIndex))
    }
}

impl Add<ByteOffset> for SourceIndex {
    type Output = SourceIndex;

    #[inline]
    fn add(self, rhs: ByteOffset) -> Self {
        if self == Self::UNKNOWN {
            return Self::UNKNOWN;
        }
        let source = self.source_id();
        let index = self.index();
        let new_index = ByteIndex(index.0) + rhs;
        Self::new(source, new_index)
    }
}

impl AddAssign<usize> for SourceIndex {
    #[inline]
    fn add_assign(&mut self, rhs: usize) {
        *self = *self + rhs;
    }
}

impl AddAssign<ByteOffset> for SourceIndex {
    #[inline]
    fn add_assign(&mut self, rhs: ByteOffset) {
        *self = *self + rhs;
    }
}

impl Sub<usize> for SourceIndex {
    type Output = SourceIndex;

    #[inline]
    fn sub(self, rhs: usize) -> Self {
        if self == Self::UNKNOWN {
            return Self::UNKNOWN;
        }
        let source = self.source_id();
        let index = self.index();
        let new_index = index.0 as RawOffset - rhs as RawOffset;
        Self::new(source, ByteIndex(new_index as RawIndex))
    }
}

impl SubAssign<usize> for SourceIndex {
    #[inline]
    fn sub_assign(&mut self, rhs: usize) {
        *self = *self - rhs;
    }
}
