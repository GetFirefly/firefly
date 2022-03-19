use std::cmp::{Ord, Ordering, PartialOrd};
use std::hash::{Hash, Hasher};

use num_traits::{CheckedShl, CheckedShr};

use super::{BitCarrier, BitRead, BitTransport, BitWrite};

/// Changes the alignment/length of a bit carrier.
#[derive(Debug, Clone)]
pub struct BitSlice<I> {
    inner: I,
    word_offset: usize,
    bit_offset: u32,
    bit_length: usize,
}
impl<I, Tr> BitCarrier for BitSlice<I>
where
    I: BitCarrier<T = Tr>,
    Tr: BitTransport,
{
    type T = Tr;
    fn bit_len(&self) -> usize {
        self.bit_length
    }
}
impl<I> BitRead for BitSlice<I>
where
    I: BitRead,
{
    fn read_word(&self, n: usize) -> Self::T {
        let p1 = self.inner.read_word(self.word_offset + n + 0);
        let p2 = self.inner.read_word(self.word_offset + n + 1);
        (p1.checked_shl(self.bit_offset).unwrap_or(Self::T::ZERO))
            | (p2
                .checked_shr(Self::T::BIT_SIZE as u32 - self.bit_offset)
                .unwrap_or(Self::T::ZERO))
    }
}

impl<I> BitWrite for BitSlice<I>
where
    I: BitWrite,
{
    fn write_word(&mut self, n: usize, data: Self::T, mask: Self::T) {
        // TODO find a way to avoid branch?
        if mask == Self::T::ZERO {
            return;
        }
        if self.bit_offset == 0 {
            self.inner.write_word(self.word_offset + n, data, mask);
        } else {
            self.inner.write_word(
                self.word_offset + n,
                data >> self.bit_offset,
                mask >> self.bit_offset,
            );
            self.inner.write_word(
                self.word_offset + n + 1,
                data << (Self::T::BIT_SIZE as u32 - self.bit_offset),
                mask << (Self::T::BIT_SIZE as u32 - self.bit_offset),
            );
        }
    }
}

impl<I> BitSlice<I>
where
    I: BitCarrier,
{
    pub fn with_offset_length(inner: I, bit_offset: usize, bit_len: usize) -> Self {
        assert!(bit_offset + bit_len <= inner.bit_len());
        BitSlice {
            inner,
            word_offset: bit_offset / I::T::BIT_SIZE,
            bit_offset: (bit_offset % I::T::BIT_SIZE) as u32,
            bit_length: bit_len,
        }
    }
}

impl<I> BitSlice<I>
where
    I: BitRead,
{
    // Iterates over each underlying transport item in the slice
    //pub fn word_iter(&self) -> BitSliceWordIter<I> {
    //    BitSliceWordIter {
    //        slice: self,
    //        idx: 0,
    //        rem: self.bit_length + 8,
    //    }
    //}
}

impl<I> Eq for BitSlice<I> where I: BitRead {}
impl<I, O, T> PartialEq<O> for BitSlice<I>
where
    I: BitRead<T = T>,
    O: BitRead<T = T>,
    T: BitTransport,
{
    fn eq(&self, other: &O) -> bool {
        if self.bit_len() != other.bit_len() {
            return false;
        }
        self.iter_words().eq(other.iter_words())
    }
}

impl<I, O, T> PartialOrd<O> for BitSlice<I>
where
    I: BitRead<T = T>,
    O: BitRead<T = T>,
    T: BitTransport,
{
    fn partial_cmp(&self, other: &O) -> Option<Ordering> {
        match self.iter_words().cmp(other.iter_words()) {
            Ordering::Equal => (),
            non_eq => return Some(non_eq),
        }
        Some(self.bit_len().cmp(&other.bit_len()))
    }
}
impl<I> Ord for BitSlice<I>
where
    I: BitRead,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl<I> Hash for BitSlice<I>
where
    I: BitRead,
{
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.bit_len().hash(state);
        for elem in self.iter_words() {
            elem.hash(state);
        }
    }
}
