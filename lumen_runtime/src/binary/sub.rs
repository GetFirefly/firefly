use std::cmp::Ordering;
use std::convert::{TryFrom, TryInto};
use std::iter::FusedIterator;

use crate::atom::{self, Existence};
use crate::binary::{heap, Part};
use crate::integer::Integer;
use crate::process::{IntoProcess, OrderInProcess, Process};
use crate::term::{BadArgument, Tag, Term};

pub struct Binary {
    #[allow(dead_code)]
    header: Term,
    pub original: Term,
    pub byte_offset: usize,
    pub bit_offset: u8,
    pub byte_count: usize,
    pub bit_count: u8,
}

impl Binary {
    pub fn new(
        original: Term,
        byte_offset: usize,
        bit_offset: u8,
        byte_count: usize,
        bit_count: u8,
    ) -> Self {
        assert_eq!(original.tag(), Tag::Boxed);

        let unboxed: &Term = original.unbox_reference();
        let unboxed_tag = unboxed.tag();

        assert!(
            (unboxed_tag == Tag::HeapBinary) | (unboxed_tag == Tag::ReferenceCountedBinary),
            "Unbox original ({:#b}) is tagged ({:?}) neither as heap or reference counted binary",
            unboxed.tagged,
            unboxed_tag
        );

        Binary {
            header: Term {
                tagged: Tag::Subbinary as usize,
            },
            original,
            byte_offset,
            byte_count,
            bit_offset,
            bit_count,
        }
    }

    /// Iterator of the [bit_count] bits.  To get the [byte_count] bytes at the beginning of the
    /// bitstring use [byte_iter].
    pub fn bit_iter(&self) -> BitIter {
        BitIter {
            original: self.original,
            byte_offset: self.byte_offset + (self.bit_count as usize),
            bit_offset: self.bit_offset,
            current_bit_count: 0,
            max_bit_count: self.bit_count,
        }
    }

    /// Iterator for the [byte_count] bytes.  For the [bit_count] bits in the partial byte at the
    /// end, use [byte_iter].
    pub fn byte_iter(&self) -> ByteIter {
        ByteIter {
            original: self.original,
            byte_offset: self.byte_offset,
            bit_offset: self.bit_offset,
            current_byte_count: 0,
            max_byte_count: self.byte_count,
        }
    }

    pub fn is_binary(&self) -> bool {
        self.bit_count == 0
    }

    /// The [byte_count] as `size` works on all binaries.
    pub fn size(&self) -> Integer {
        self.byte_count.into()
    }

    /// Converts to atom only if [bit_count] is `0`.
    pub fn to_atom_index(
        &self,
        existence: Existence,
        process: &mut Process,
    ) -> Result<atom::Index, BadArgument> {
        let string: String = self.try_into()?;

        process.str_to_atom_index(&string, existence)
    }

    pub fn to_list(&self, mut process: &mut Process) -> Result<Term, BadArgument> {
        if self.bit_count == 0 {
            let list = self.byte_iter().rfold(Term::EMPTY_LIST, |acc, byte| {
                Term::cons(byte.into_process(&mut process), acc, &mut process)
            });

            Ok(list)
        } else {
            Err(BadArgument)
        }
    }
}

impl TryFrom<&Binary> for Vec<u8> {
    type Error = BadArgument;

    fn try_from(binary: &Binary) -> Result<Vec<u8>, BadArgument> {
        if 0 < binary.bit_count {
            Err(BadArgument)
        } else {
            let mut bytes_vec: Vec<u8> = Vec::with_capacity(binary.byte_count);
            bytes_vec.extend(binary.byte_iter());

            Ok(bytes_vec)
        }
    }
}

impl TryFrom<&Binary> for String {
    type Error = BadArgument;

    fn try_from(binary: &Binary) -> Result<String, BadArgument> {
        let byte_vec: Vec<u8> = binary.try_into()?;

        String::from_utf8(byte_vec).map_err(|_| BadArgument)
    }
}

pub struct BitIter {
    original: Term,
    byte_offset: usize,
    bit_offset: u8,
    current_bit_count: u8,
    max_bit_count: u8,
}

pub struct ByteIter {
    original: Term,
    byte_offset: usize,
    bit_offset: u8,
    current_byte_count: usize,
    max_byte_count: usize,
}

impl ByteIter {
    fn byte(&self, index: usize) -> u8 {
        let first_index = self.byte_offset + index;
        let first_byte = self.original.byte(first_index);

        if 0 < self.bit_offset {
            let second_byte = self.original.byte(first_index + 1);

            (first_byte << self.bit_offset) | (second_byte >> (8 - self.bit_offset))
        } else {
            first_byte
        }
    }
}

impl Iterator for BitIter {
    type Item = u8;

    fn next(&mut self) -> Option<u8> {
        if self.current_bit_count == self.max_bit_count {
            None
        } else {
            let first_index = self.byte_offset;
            let first_original_byte = self.original.byte(first_index);

            let byte = if 0 < self.bit_offset {
                let second_original_byte = self.original.byte(first_index + 1);
                (first_original_byte << self.bit_offset)
                    | (second_original_byte >> (8 - self.bit_offset))
            } else {
                first_original_byte
            };

            let bit = (byte >> (7 - self.current_bit_count)) & 0b1;

            self.current_bit_count += 1;

            Some(bit)
        }
    }
}

impl Iterator for ByteIter {
    type Item = u8;

    fn next(&mut self) -> Option<u8> {
        if self.current_byte_count == self.max_byte_count {
            None
        } else {
            let byte = self.byte(self.current_byte_count);
            self.current_byte_count += 1;

            Some(byte)
        }
    }
}

impl DoubleEndedIterator for ByteIter {
    fn next_back(&mut self) -> Option<u8> {
        if self.current_byte_count == self.max_byte_count {
            None
        } else {
            self.max_byte_count -= 1;
            let byte = self.byte(self.max_byte_count);

            Some(byte)
        }
    }
}

impl FusedIterator for ByteIter {}

impl OrderInProcess<heap::Binary> for Binary {
    /// > * Bitstrings are compared byte by byte, incomplete bytes are compared bit by bit.
    /// > -- https://hexdocs.pm/elixir/operators.html#term-ordering
    fn cmp_in_process(&self, other: &heap::Binary, _process: &Process) -> Ordering {
        match self.byte_iter().cmp(other.byte_iter()) {
            Ordering::Equal =>
            // a heap::Binary has 0 bit_count, so if the subbinary has any tail bits it is greater
            {
                if self.bit_count > 0 {
                    Ordering::Greater
                } else {
                    Ordering::Equal
                }
            }
            ordering => ordering,
        }
    }
}

impl OrderInProcess<Binary> for Binary {
    /// > * Bitstrings are compared byte by byte, incomplete bytes are compared bit by bit.
    /// > -- https://hexdocs.pm/elixir/operators.html#term-ordering
    fn cmp_in_process(&self, other: &Binary, _process: &Process) -> Ordering {
        match self.byte_iter().cmp(other.byte_iter()) {
            Ordering::Equal => self.bit_iter().cmp(other.bit_iter()),
            ordering => ordering,
        }
    }
}

impl<'b, 'a: 'b> Part<'a, usize, isize, &'b Binary> for Binary {
    fn part(
        &'a self,
        start: usize,
        length: isize,
        process: &mut Process,
    ) -> Result<&'b Binary, BadArgument> {
        let byte_count_isize = self.byte_count as isize;

        // new subbinary is entire subbinary
        if (self.bit_count == 0)
            & (((start == 0) & (length == byte_count_isize))
                | ((start == self.byte_count) & (length == -byte_count_isize)))
        {
            Ok(self)
        } else if length >= 0 {
            let non_negative_length = length as usize;

            if (start < self.byte_count) & (start + non_negative_length <= self.byte_count) {
                let new_subbinary = process.subbinary(
                    self.original,
                    self.byte_offset + start,
                    self.bit_offset,
                    non_negative_length,
                    0,
                );
                Ok(new_subbinary)
            } else {
                Err(BadArgument)
            }
        } else {
            let start_isize = start as isize;

            if (start <= self.byte_count) & (0 <= start_isize + length) {
                let byte_offset = (start_isize + length) as usize;
                let byte_count = (-length) as usize;
                let new_subbinary =
                    process.subbinary(self.original, byte_offset, self.bit_offset, byte_count, 0);

                Ok(new_subbinary)
            } else {
                Err(BadArgument)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod byte_iter {
        use super::*;

        #[test]
        fn is_double_ended() {
            let mut process: Process = Default::default();
            // <<1::1, 0, 1, 2>>
            let binary = Term::slice_to_binary(&[128, 0, 129, 0b0000_0000], &mut process);
            let subbinary = Binary::new(binary, 0, 1, 3, 0);

            let mut iter = subbinary.byte_iter();

            assert_eq!(iter.next(), Some(0));
            assert_eq!(iter.next(), Some(1));
            assert_eq!(iter.next(), Some(2));
            assert_eq!(iter.next(), None);
            assert_eq!(iter.next(), None);

            let mut rev_iter = subbinary.byte_iter();

            assert_eq!(rev_iter.next_back(), Some(2));
            assert_eq!(rev_iter.next_back(), Some(1));
            assert_eq!(rev_iter.next_back(), Some(0));
            assert_eq!(rev_iter.next_back(), None);
            assert_eq!(rev_iter.next_back(), None);

            let mut double_ended_iter = subbinary.byte_iter();

            assert_eq!(double_ended_iter.next(), Some(0));
            assert_eq!(double_ended_iter.next_back(), Some(2));
            assert_eq!(double_ended_iter.next(), Some(1));
            assert_eq!(double_ended_iter.next_back(), None);
            assert_eq!(double_ended_iter.next(), None);
        }
    }
}
