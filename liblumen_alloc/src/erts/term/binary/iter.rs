use core::iter::{Copied, FusedIterator};
use core::slice;

use crate::erts::term::prelude::Boxed;

use super::IndexByte;

/// This trait provides tools common to all binary iterators which iterator by bytes
pub trait ByteIterator<'a>: ExactSizeIterator + DoubleEndedIterator + Iterator<Item = u8> {}
// By default, byte slices implement `ByteIterator`
impl<'a> ByteIterator<'a> for Copied<slice::Iter<'a, u8>> {}

/// This trait provides tools common to all binary iterators which iterate by bits
pub trait BitIterator: Iterator<Item = u8> {}

/// This iterable iterates over the bytes between a pair of offsets in a given binary
#[derive(Debug)]
pub struct FullByteIter<'a, T: ?Sized + IndexByte> {
    bin: &'a T,
    base_byte_offset: usize,
    bit_offset: u8,
    current_byte_offset: usize,
    max_byte_offset: usize,
}

impl<'a, T: ?Sized + IndexByte> FullByteIter<'a, T> {
    pub fn new(bin: Boxed<T>, base_byte_offset: usize, bit_offset: u8, current_byte_offset: usize, max_byte_offset: usize) -> Self {
        Self {
            bin: bin.as_ref(),
            base_byte_offset,
            bit_offset,
            current_byte_offset,
            max_byte_offset,
        }
    }

    fn is_aligned(&self) -> bool {
        self.bit_offset == 0
    }

    fn byte(&self, index: usize) -> u8 {
        let first_index = self.base_byte_offset + index;
        let first_byte = self.bin.byte(first_index);

        if self.is_aligned() {
            first_byte
        } else {
            let second_byte = self.bin.byte(first_index + 1);
            (first_byte << self.bit_offset) | (second_byte >> (8 - self.bit_offset))
        }
    }
}
impl<'a, T> ByteIterator<'a> for FullByteIter<'a, T> where T: ?Sized + IndexByte {}
impl<'a, T> ExactSizeIterator for FullByteIter<'a, T> where T: ?Sized + IndexByte {}
impl<'a, T> FusedIterator for FullByteIter<'a, T> where T: ?Sized + IndexByte {}
impl<'a, T> DoubleEndedIterator for FullByteIter<'a, T>
where
    T: ?Sized + IndexByte,
{
    fn next_back(&mut self) -> Option<u8> {
        if self.current_byte_offset == self.max_byte_offset {
            None
        } else {
            self.max_byte_offset -= 1;
            let byte = self.byte(self.max_byte_offset);

            Some(byte)
        }
    }
}
impl<'a, T> Iterator for FullByteIter<'a, T>
where
    T: ?Sized + IndexByte,
{
    type Item = u8;

    fn next(&mut self) -> Option<u8> {
        if self.current_byte_offset == self.max_byte_offset {
            None
        } else {
            let byte = self.byte(self.current_byte_offset);
            self.current_byte_offset += 1;

            Some(byte)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = self.max_byte_offset - self.current_byte_offset;

        (size, Some(size))
    }
}

/// This iterable iterates over the bits in a byte, up to some maximum number of bits
#[derive(Debug)]
pub struct BitsIter {
    byte: u8,
    current_bit_offset: u8,
    max_bit_offset: u8,
}
impl BitsIter {
    #[inline]
    pub fn new(byte: u8) -> Self {
        Self::new_with_max(byte, 8)
    }

    #[inline]
    pub fn new_with_max(byte: u8, bit_len: u8) -> Self {
        Self {
            byte,
            current_bit_offset: 0,
            max_bit_offset: bit_len,
        }
    }
}
impl Iterator for BitsIter {
    type Item = u8;

    fn next(&mut self) -> Option<u8> {
        if self.current_bit_offset == self.max_bit_offset {
            None
        } else {
            let bit = (self.byte >> (7 - self.current_bit_offset)) & 0b1;

            self.current_bit_offset += 1;

            Some(bit)
        }
    }
}

/// This iterable iterates over the bits between a pair of byte/bit offsets
#[derive(Debug)]
pub struct PartialByteBitIter<'a, T: ?Sized + IndexByte> {
    bin: &'a T,
    current_byte_offset: usize,
    current_bit_offset: u8,
    max_byte_offset: usize,
    max_bit_offset: u8,
}
impl<'a, T: ?Sized + IndexByte> PartialByteBitIter<'a, T> {
    #[inline]
    pub fn new(bin: Boxed<T>, current_byte_offset: usize, current_bit_offset: u8, max_byte_offset: usize, max_bit_offset: u8) -> Self {
        Self {
            bin: bin.as_ref(),
            current_byte_offset,
            current_bit_offset,
            max_byte_offset,
            max_bit_offset,
        }
    }
}
impl<'a, T> BitIterator for PartialByteBitIter<'a, T> where T: ?Sized + IndexByte {}
impl<'a, T> Iterator for PartialByteBitIter<'a, T>
where
    T: ?Sized + IndexByte,
{
    type Item = u8;

    fn next(&mut self) -> Option<u8> {
        if (self.current_byte_offset == self.max_byte_offset)
            & (self.current_bit_offset == self.max_bit_offset)
        {
            None
        } else {
            let byte = self.bin.byte(self.current_byte_offset);
            let bit = (byte >> (7 - self.current_bit_offset)) & 0b1;

            if self.current_bit_offset == 7 {
                self.current_bit_offset = 0;
                self.current_byte_offset += 1;
            } else {
                self.current_bit_offset += 1;
            }

            Some(bit)
        }
    }
}
