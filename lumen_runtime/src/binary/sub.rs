use std::cmp::Ordering::{self, *};
use std::convert::{TryFrom, TryInto};
use std::fmt::{self, Debug};
use std::hash::{Hash, Hasher};
use std::iter::FusedIterator;

use crate::atom::{self, Existence};
use crate::binary::{
    heap, part_range_to_list, start_length_to_part_range, ByteIterator, Part, PartRange,
    PartToList, ToTerm, ToTermOptions,
};
use crate::exception::{self, Exception};
use crate::integer::Integer;
use crate::process::{IntoProcess, Process};
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
                        let original_byte_count = heap_binary.byte_len();
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
        let current_byte_offset = self.byte_offset + (self.byte_count as usize);
        let current_bit_offset = self.bit_offset;

        let improper_bit_offset = current_bit_offset + self.bit_count;
        let max_byte_offset = current_byte_offset + (improper_bit_offset / 8) as usize;
        let max_bit_offset = improper_bit_offset % 8;

        BitCountIter {
            original: self.original,
            current_byte_offset,
            current_bit_offset,
            max_byte_offset,
            max_bit_offset,
        }
    }

    /// The total number of bits including bits in [byte_count] and [bit_count].
    pub fn bit_len(&self) -> usize {
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

    pub fn byte_len(&self) -> usize {
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
    pub fn to_atom_index(&self, existence: Existence) -> Result<atom::Index, Exception> {
        let string: String = self.try_into()?;

        atom::str_to_index(&string, existence).ok_or_else(|| badarg!())
    }

    pub fn to_list(&self, mut process: &mut Process) -> exception::Result {
        if self.bit_count == 0 {
            let list = self.byte_iter().rfold(Term::EMPTY_LIST, |acc, byte| {
                Term::cons(byte.into_process(&mut process), acc, &mut process)
            });

            Ok(list)
        } else {
            Err(badarg!())
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

impl Debug for Binary {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Binary::new({:?}, {:?}, {:?}, {:?}, {:?})",
            self.original, self.byte_offset, self.bit_offset, self.byte_count, self.bit_count
        )
    }
}

impl Eq for Binary {}

impl Hash for Binary {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for byte in self.byte_iter() {
            byte.hash(state);
        }

        for bit in self.bit_count_iter() {
            bit.hash(state);
        }
    }
}

impl PartialEq for Binary {
    fn eq(&self, other: &Binary) -> bool {
        (self.bit_len() == other.bit_len())
            & self
                .byte_iter()
                .zip(other.byte_iter())
                .all(|(self_byte, other_byte)| self_byte == other_byte)
            & self
                .bit_count_iter()
                .zip(other.bit_count_iter())
                .all(|(self_bit, other_bit)| self_bit == other_bit)
    }
}

impl ToTerm for Binary {
    fn to_term(&self, options: ToTermOptions, mut process: &mut Process) -> exception::Result {
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
                None => Err(badarg!()),
            }
        } else {
            Err(badarg!())
        }
    }
}

impl TryFrom<&Binary> for Vec<u8> {
    type Error = Exception;

    fn try_from(binary: &Binary) -> Result<Vec<u8>, Exception> {
        if 0 < binary.bit_count {
            Err(badarg!())
        } else {
            let mut bytes_vec: Vec<u8> = Vec::with_capacity(binary.byte_count);
            bytes_vec.extend(binary.byte_iter());

            Ok(bytes_vec)
        }
    }
}

impl TryFrom<&Binary> for String {
    type Error = Exception;

    fn try_from(binary: &Binary) -> Result<String, Exception> {
        let byte_vec: Vec<u8> = binary.try_into()?;

        String::from_utf8(byte_vec).map_err(|_| badarg!())
    }
}

pub struct BitCountIter {
    original: Term,
    current_byte_offset: usize,
    current_bit_offset: u8,
    max_byte_offset: usize,
    max_bit_offset: u8,
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
        if (self.current_byte_offset == self.max_byte_offset)
            & (self.current_bit_offset == self.max_bit_offset)
        {
            None
        } else {
            let byte = self.original.byte(self.current_byte_offset);
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

impl Ord for Binary {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl PartialEq<heap::Binary> for Binary {
    /// > * Bitstrings are compared byte by byte, incomplete bytes are compared bit by bit.
    /// > -- https://hexdocs.pm/elixir/operators.html#term-ordering
    fn eq(&self, other: &heap::Binary) -> bool {
        (self.bit_count == 0) & self.byte_iter().eq(other.byte_iter())
    }
}

impl PartialOrd<heap::Binary> for Binary {
    /// > * Bitstrings are compared byte by byte, incomplete bytes are compared bit by bit.
    /// > -- https://hexdocs.pm/elixir/operators.html#term-ordering
    fn partial_cmp(&self, other: &heap::Binary) -> Option<Ordering> {
        match self.byte_iter().partial_cmp(other.byte_iter()) {
            Some(Equal) =>
            // a heap::Binary has 0 bit_count, so if the subbinary has any tail bits it is greater
            {
                if 0 < self.bit_count {
                    Some(Greater)
                } else {
                    Some(Equal)
                }
            }
            partial_ordering => partial_ordering,
        }
    }
}

impl PartialOrd<Binary> for Binary {
    /// > * Bitstrings are compared byte by byte, incomplete bytes are compared bit by bit.
    /// > -- https://hexdocs.pm/elixir/operators.html#term-ordering
    fn partial_cmp(&self, other: &Binary) -> Option<Ordering> {
        match self.byte_iter().partial_cmp(other.byte_iter()) {
            Some(Equal) => self.bit_count_iter().partial_cmp(other.bit_count_iter()),
            partial_ordering => partial_ordering,
        }
    }
}

impl<'b, 'a: 'b> Part<'a, usize, isize, &'b Binary> for Binary {
    fn part(
        &'a self,
        start: usize,
        length: isize,
        process: &mut Process,
    ) -> Result<&'b Binary, Exception> {
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
    ) -> Result<Term, Exception> {
        let part_range = start_length_to_part_range(start, length, self.byte_count)?;
        let list = part_range_to_list(self.byte_iter(), part_range, &mut process);

        Ok(list)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod bit_count_iter {
        use super::*;

        use std::sync::{Arc, RwLock};

        use crate::environment::{self, Environment};

        mod with_0_bit_offset {
            use super::*;

            #[test]
            fn with_0_bit_count() {
                let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
                let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
                let mut process = process_rw_lock.write().unwrap();
                let binary = Term::slice_to_binary(&[0b1111_1111], &mut process);
                let subbinary = Binary::new(binary, 0, 0, 1, 0);

                let mut bit_count_iter = subbinary.bit_count_iter();

                assert_eq!(bit_count_iter.next(), None);
            }

            #[test]
            fn with_1_bit_count() {
                let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
                let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
                let mut process = process_rw_lock.write().unwrap();
                let binary = Term::slice_to_binary(&[0b1111_1111, 0b1000_0000], &mut process);
                let subbinary = Binary::new(binary, 0, 0, 1, 1);

                let mut bit_count_iter = subbinary.bit_count_iter();

                assert_eq!(bit_count_iter.next(), Some(1));
                assert_eq!(bit_count_iter.next(), None);
            }

            #[test]
            fn with_2_bit_count() {
                let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
                let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
                let mut process = process_rw_lock.write().unwrap();
                let binary = Term::slice_to_binary(&[0b1111_1111, 0b1100_0000], &mut process);
                let subbinary = Binary::new(binary, 0, 0, 1, 2);

                let mut bit_count_iter = subbinary.bit_count_iter();

                assert_eq!(bit_count_iter.next(), Some(1));
                assert_eq!(bit_count_iter.next(), Some(1));
                assert_eq!(bit_count_iter.next(), None);
            }

            #[test]
            fn with_3_bit_count() {
                let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
                let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
                let mut process = process_rw_lock.write().unwrap();
                let binary = Term::slice_to_binary(&[0b1111_1111, 0b1110_0000], &mut process);
                let subbinary = Binary::new(binary, 0, 0, 1, 3);

                let mut bit_count_iter = subbinary.bit_count_iter();

                assert_eq!(bit_count_iter.next(), Some(1));
                assert_eq!(bit_count_iter.next(), Some(1));
                assert_eq!(bit_count_iter.next(), Some(1));
                assert_eq!(bit_count_iter.next(), None);
            }

            #[test]
            fn with_4_bit_count() {
                let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
                let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
                let mut process = process_rw_lock.write().unwrap();
                let binary = Term::slice_to_binary(&[0b1111_1111, 0b1111_0000], &mut process);
                let subbinary = Binary::new(binary, 0, 0, 1, 4);

                let mut bit_count_iter = subbinary.bit_count_iter();

                assert_eq!(bit_count_iter.next(), Some(1));
                assert_eq!(bit_count_iter.next(), Some(1));
                assert_eq!(bit_count_iter.next(), Some(1));
                assert_eq!(bit_count_iter.next(), Some(1));
                assert_eq!(bit_count_iter.next(), None);
            }

            #[test]
            fn with_5_bit_count() {
                let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
                let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
                let mut process = process_rw_lock.write().unwrap();
                let binary = Term::slice_to_binary(&[0b1111_1111, 0b1111_1000], &mut process);
                let subbinary = Binary::new(binary, 0, 0, 1, 5);

                let mut bit_count_iter = subbinary.bit_count_iter();

                assert_eq!(bit_count_iter.next(), Some(1));
                assert_eq!(bit_count_iter.next(), Some(1));
                assert_eq!(bit_count_iter.next(), Some(1));
                assert_eq!(bit_count_iter.next(), Some(1));
                assert_eq!(bit_count_iter.next(), Some(1));
                assert_eq!(bit_count_iter.next(), None);
            }

            #[test]
            fn with_6_bit_count() {
                let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
                let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
                let mut process = process_rw_lock.write().unwrap();
                let binary = Term::slice_to_binary(&[0b1111_1111, 0b1111_1100], &mut process);
                let subbinary = Binary::new(binary, 0, 0, 1, 6);

                let mut bit_count_iter = subbinary.bit_count_iter();

                assert_eq!(bit_count_iter.next(), Some(1));
                assert_eq!(bit_count_iter.next(), Some(1));
                assert_eq!(bit_count_iter.next(), Some(1));
                assert_eq!(bit_count_iter.next(), Some(1));
                assert_eq!(bit_count_iter.next(), Some(1));
                assert_eq!(bit_count_iter.next(), Some(1));
                assert_eq!(bit_count_iter.next(), None);
            }

            #[test]
            fn with_7_bit_count() {
                let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
                let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
                let mut process = process_rw_lock.write().unwrap();
                let binary = Term::slice_to_binary(&[0b1111_1111, 0b1111_1110], &mut process);
                let subbinary = Binary::new(binary, 0, 0, 1, 7);

                let mut bit_count_iter = subbinary.bit_count_iter();

                assert_eq!(bit_count_iter.next(), Some(1));
                assert_eq!(bit_count_iter.next(), Some(1));
                assert_eq!(bit_count_iter.next(), Some(1));
                assert_eq!(bit_count_iter.next(), Some(1));
                assert_eq!(bit_count_iter.next(), Some(1));
                assert_eq!(bit_count_iter.next(), Some(1));
                assert_eq!(bit_count_iter.next(), Some(1));
                assert_eq!(bit_count_iter.next(), None);
            }
        }

        mod with_1_bit_offset {
            use super::*;

            #[test]
            fn with_0_bit_count() {
                let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
                let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
                let mut process = process_rw_lock.write().unwrap();
                let binary = Term::slice_to_binary(&[0b1111_1111, 0b1000_0000], &mut process);
                let subbinary = Binary::new(binary, 0, 1, 1, 0);

                let mut bit_count_iter = subbinary.bit_count_iter();

                assert_eq!(bit_count_iter.next(), None);
            }

            #[test]
            fn with_1_bit_count() {
                let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
                let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
                let mut process = process_rw_lock.write().unwrap();
                let binary = Term::slice_to_binary(&[0b1111_1111, 0b1100_0000], &mut process);
                let subbinary = Binary::new(binary, 0, 1, 1, 1);

                let mut bit_count_iter = subbinary.bit_count_iter();

                assert_eq!(bit_count_iter.next(), Some(1));
                assert_eq!(bit_count_iter.next(), None);
            }

            #[test]
            fn with_2_bit_count() {
                let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
                let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
                let mut process = process_rw_lock.write().unwrap();
                let binary = Term::slice_to_binary(&[0b1111_1111, 0b1110_0000], &mut process);
                let subbinary = Binary::new(binary, 0, 1, 1, 2);

                let mut bit_count_iter = subbinary.bit_count_iter();

                assert_eq!(bit_count_iter.next(), Some(1));
                assert_eq!(bit_count_iter.next(), Some(1));
                assert_eq!(bit_count_iter.next(), None);
            }

            #[test]
            fn with_3_bit_count() {
                let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
                let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
                let mut process = process_rw_lock.write().unwrap();
                let binary = Term::slice_to_binary(&[0b1111_1111, 0b1111_0000], &mut process);
                let subbinary = Binary::new(binary, 0, 1, 1, 3);

                let mut bit_count_iter = subbinary.bit_count_iter();

                assert_eq!(bit_count_iter.next(), Some(1));
                assert_eq!(bit_count_iter.next(), Some(1));
                assert_eq!(bit_count_iter.next(), Some(1));
                assert_eq!(bit_count_iter.next(), None);
            }

            #[test]
            fn with_4_bit_count() {
                let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
                let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
                let mut process = process_rw_lock.write().unwrap();
                let binary = Term::slice_to_binary(&[0b1111_1111, 0b1111_1000], &mut process);
                let subbinary = Binary::new(binary, 0, 1, 1, 4);

                let mut bit_count_iter = subbinary.bit_count_iter();

                assert_eq!(bit_count_iter.next(), Some(1));
                assert_eq!(bit_count_iter.next(), Some(1));
                assert_eq!(bit_count_iter.next(), Some(1));
                assert_eq!(bit_count_iter.next(), Some(1));
                assert_eq!(bit_count_iter.next(), None);
            }

            #[test]
            fn with_5_bit_count() {
                let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
                let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
                let mut process = process_rw_lock.write().unwrap();
                let binary = Term::slice_to_binary(&[0b1111_1111, 0b1111_1100], &mut process);
                let subbinary = Binary::new(binary, 0, 1, 1, 5);

                let mut bit_count_iter = subbinary.bit_count_iter();

                assert_eq!(bit_count_iter.next(), Some(1));
                assert_eq!(bit_count_iter.next(), Some(1));
                assert_eq!(bit_count_iter.next(), Some(1));
                assert_eq!(bit_count_iter.next(), Some(1));
                assert_eq!(bit_count_iter.next(), Some(1));
                assert_eq!(bit_count_iter.next(), None);
            }

            #[test]
            fn with_6_bit_count() {
                let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
                let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
                let mut process = process_rw_lock.write().unwrap();
                let binary = Term::slice_to_binary(&[0b1111_1111, 0b1111_1110], &mut process);
                let subbinary = Binary::new(binary, 0, 1, 1, 6);

                let mut bit_count_iter = subbinary.bit_count_iter();

                assert_eq!(bit_count_iter.next(), Some(1));
                assert_eq!(bit_count_iter.next(), Some(1));
                assert_eq!(bit_count_iter.next(), Some(1));
                assert_eq!(bit_count_iter.next(), Some(1));
                assert_eq!(bit_count_iter.next(), Some(1));
                assert_eq!(bit_count_iter.next(), Some(1));
                assert_eq!(bit_count_iter.next(), None);
            }

            #[test]
            fn with_7_bit_count() {
                let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
                let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
                let mut process = process_rw_lock.write().unwrap();
                let binary = Term::slice_to_binary(&[0b1111_1111, 0b1111_1111], &mut process);
                let subbinary = Binary::new(binary, 0, 1, 1, 7);

                let mut bit_count_iter = subbinary.bit_count_iter();

                assert_eq!(bit_count_iter.next(), Some(1));
                assert_eq!(bit_count_iter.next(), Some(1));
                assert_eq!(bit_count_iter.next(), Some(1));
                assert_eq!(bit_count_iter.next(), Some(1));
                assert_eq!(bit_count_iter.next(), Some(1));
                assert_eq!(bit_count_iter.next(), Some(1));
                assert_eq!(bit_count_iter.next(), Some(1));
                assert_eq!(bit_count_iter.next(), None);
            }
        }
    }

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
