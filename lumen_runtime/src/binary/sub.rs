use std::cmp::Ordering;
use std::convert::{TryFrom, TryInto};
use std::iter::FusedIterator;

use crate::atom::{self, Existence};
use crate::bad_argument::BadArgument;
use crate::binary::{
    heap, part_range_to_list, start_length_to_part_range, ByteIterator, Part, PartRange,
    PartToList, ToTerm, ToTermOptions,
};
use crate::integer::Integer;
use crate::process::{IntoProcess, OrderInProcess, Process};
use crate::term::{Tag::*, Term};

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
        match original.tag() {
            Boxed => {
                let unboxed: &Term = original.unbox_reference();

                match unboxed.tag() {
                    HeapBinary => {
                        let heap_binary: &heap::Binary = original.unbox_reference();
                        let original_byte_count = heap_binary.byte_size();
                        let original_bit_count = original_byte_count * 8;
                        let required_bit_count = byte_offset * 8
                            + (bit_offset as usize)
                            + 8 * byte_count
                            + (bit_count as usize);

                        assert!(
                            required_bit_count <= original_bit_count,
                            "Required bit count ({}) is greater than original bit count ({})",
                            required_bit_count,
                            original_bit_count
                        );
                    }
                    unboxed_tag => panic!(
                        "Unboxed tag ({:?}) cannot be original binary for subbinary",
                        unboxed_tag
                    ),
                }
            }
            tag => panic!("Tag ({:?}) cannot be original binary for subbinary", tag),
        }

        Binary {
            header: Term {
                tagged: Subbinary as usize,
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
    pub fn bit_count_iter(&self) -> BitCountIter {
        BitCountIter {
            original: self.original,
            byte_offset: self.byte_offset + (self.byte_count as usize),
            bit_offset: self.bit_offset,
            current_bit_count: 0,
            max_bit_count: self.bit_count,
        }
    }

    /// The total number of bits including bits in [byte_count] and [bit_count].
    pub fn bit_size(&self) -> usize {
        self.byte_count * 8 + (self.bit_count as usize)
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

    pub fn byte_size(&self) -> usize {
        self.byte_count + if 0 < self.bit_count { 1 } else { 0 }
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
            Err(bad_argument!())
        }
    }

    pub fn to_bitstring_list(&self, mut process: &mut Process) -> Term {
        let initial = if self.bit_count == 0 {
            Term::EMPTY_LIST
        } else {
            self.bit_count_subbinary(&mut process)
        };

        self.byte_iter().rfold(initial, |acc, byte| {
            Term::cons(byte.into_process(&mut process), acc, &mut process)
        })
    }

    fn bit_count_subbinary(&self, mut process: &mut Process) -> Term {
        Term::subbinary(
            self.original,
            self.byte_offset + (self.byte_count as usize),
            self.bit_offset,
            0,
            self.bit_count,
            &mut process,
        )
    }
}

impl ToTerm for Binary {
    fn to_term(
        &self,
        options: ToTermOptions,
        mut process: &mut Process,
    ) -> Result<Term, BadArgument> {
        if self.bit_count == 0 {
            let mut byte_iter = self.byte_iter();

            match byte_iter.next_versioned_term(options.existence, &mut process) {
                Some(term) => {
                    if options.used {
                        let used = self.byte_count - byte_iter.len();
                        let used_term: Term = used.into_process(&mut process);

                        Ok(Term::slice_to_tuple(&[term, used_term], &mut process))
                    } else {
                        Ok(term)
                    }
                }
                None => Err(bad_argument!()),
            }
        } else {
            Err(bad_argument!())
        }
    }
}

impl TryFrom<&Binary> for Vec<u8> {
    type Error = BadArgument;

    fn try_from(binary: &Binary) -> Result<Vec<u8>, BadArgument> {
        if 0 < binary.bit_count {
            Err(bad_argument!())
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

        String::from_utf8(byte_vec).map_err(|_| bad_argument!())
    }
}

pub struct BitCountIter {
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

impl ByteIterator for ByteIter {}

impl ExactSizeIterator for ByteIter {}

impl Iterator for BitCountIter {
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

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = self.max_byte_count - self.current_byte_count;

        (size, Some(size))
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
            Ordering::Equal => self.bit_count_iter().cmp(other.bit_count_iter()),
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
        let PartRange {
            byte_offset,
            byte_count,
        } = start_length_to_part_range(start, length, self.byte_count)?;

        // new subbinary is entire subbinary
        if (self.bit_count == 0) && (byte_offset == 0) && (byte_count == self.byte_count) {
            Ok(self)
        } else {
            let new_subbinary =
                process.subbinary(self.original, byte_offset, self.bit_offset, byte_count, 0);

            Ok(new_subbinary)
        }
    }
}

impl PartToList<usize, isize> for Binary {
    fn part_to_list(
        &self,
        start: usize,
        length: isize,
        mut process: &mut Process,
    ) -> Result<Term, BadArgument> {
        let part_range = start_length_to_part_range(start, length, self.byte_count)?;
        let list = part_range_to_list(self.byte_iter(), part_range, &mut process);

        Ok(list)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod byte_iter {
        use super::*;

        use std::sync::{Arc, RwLock};

        use crate::environment::{self, Environment};

        #[test]
        fn is_double_ended() {
            let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
            let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
            let mut process = process_rw_lock.write().unwrap();
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
