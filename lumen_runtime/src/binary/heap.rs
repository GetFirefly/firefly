use std::cmp::Ordering;
use std::convert::TryFrom;

use liblumen_arena::TypedArena;

use crate::atom::{self, Existence};
use crate::bad_argument::BadArgument;
use crate::binary::{
    self, part_range_to_list, start_length_to_part_range, ByteIterator, Part, PartRange,
    PartToList, ToTerm, ToTermOptions,
};
use crate::integer::Integer;
use crate::process::{DebugInProcess, IntoProcess, OrderInProcess, Process};
use crate::term::Term;

pub struct Binary {
    header: Term,
    bytes: *const u8,
}

impl<'binary, 'bytes: 'binary> Binary {
    pub fn from_slice(
        bytes: &[u8],
        binary_arena: &'binary mut TypedArena<Binary>,
        byte_arena: &'bytes mut TypedArena<u8>,
    ) -> &'static Self {
        let arena_bytes: &[u8] = if bytes.len() != 0 {
            byte_arena.alloc_slice(bytes)
        } else {
            &[]
        };

        let pointer = binary_arena.alloc(Binary::new(arena_bytes)) as *const Binary;

        unsafe { &*pointer }
    }

    fn new(bytes: &[u8]) -> Self {
        Binary {
            header: Term::heap_binary(bytes.len()),
            bytes: bytes.as_ptr(),
        }
    }

    pub fn bit_size(&self) -> usize {
        self.header.heap_binary_to_byte_count() * 8
    }

    pub fn byte(&self, index: usize) -> u8 {
        let byte_count = Term::heap_binary_to_byte_count(&self.header);

        assert!(
            index < byte_count,
            "index ({}) >= byte count ({})",
            index,
            byte_count
        );

        unsafe { *self.bytes.offset(index as isize) }
    }

    pub fn byte_iter(&self) -> Iter {
        self.iter()
    }

    pub fn byte_size(&self) -> usize {
        self.header.heap_binary_to_byte_count()
    }

    pub fn iter(&self) -> Iter {
        let byte_count = Term::heap_binary_to_byte_count(&self.header);

        unsafe {
            Iter {
                pointer: self.bytes,
                limit: self.bytes.offset(byte_count as isize),
            }
        }
    }

    pub fn size(&self) -> Integer {
        // The `header` field is not the same as `size` because `size` is tagged as a small integer
        // while `header` is tagged as `HeapBinary` to mark the beginning of a heap binary.
        self.header.heap_binary_to_byte_count().into()
    }

    pub fn to_atom_index(
        &self,
        existence: Existence,
        process: &mut Process,
    ) -> Result<atom::Index, BadArgument> {
        let bytes = unsafe {
            std::slice::from_raw_parts(self.bytes, Term::heap_binary_to_byte_count(&self.header))
        };

        process.str_to_atom_index(std::str::from_utf8(bytes).unwrap(), existence)
    }

    pub fn to_list(&self, mut process: &mut Process) -> Term {
        self.iter().rfold(Term::EMPTY_LIST, |acc, byte| {
            Term::cons(byte.into_process(&mut process), acc, &mut process)
        })
    }

    pub fn to_bitstring_list(&self, process: &mut Process) -> Term {
        self.to_list(process)
    }
}

impl DebugInProcess for Binary {
    fn format_in_process(&self, _process: &Process) -> String {
        let mut strings: Vec<String> = Vec::new();
        strings.push("Binary::from_slice(&[".to_string());

        let mut iter = self.iter();

        if let Some(first_byte) = iter.next() {
            strings.push(first_byte.to_string());

            for element in iter {
                strings.push(", ".to_string());
                strings.push(element.to_string());
            }
        }

        strings.push("])".to_string());
        strings.join("")
    }
}

pub struct Iter {
    pointer: *const u8,
    limit: *const u8,
}

impl ByteIterator for Iter {}

impl ExactSizeIterator for Iter {}

impl Iterator for Iter {
    type Item = u8;

    fn next(&mut self) -> Option<u8> {
        if self.pointer == self.limit {
            None
        } else {
            let old_pointer = self.pointer;

            unsafe {
                self.pointer = self.pointer.offset(1);
                old_pointer.as_ref().map(|r| *r)
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = unsafe { self.limit.offset_from(self.pointer) } as usize;

        (size, Some(size))
    }
}

impl DoubleEndedIterator for Iter {
    fn next_back(&mut self) -> Option<u8> {
        if self.pointer == self.limit {
            None
        } else {
            unsafe {
                // limit is +1 past the actual elements, so pre-decrement unlike `next`, which
                // post-decrements
                self.limit = self.limit.offset(-1);
                self.limit.as_ref().map(|r| *r)
            }
        }
    }
}

impl<'b, 'a: 'b> Part<'a, usize, isize, binary::Binary<'b>> for Binary {
    fn part(
        &'a self,
        start: usize,
        length: isize,
        process: &mut Process,
    ) -> Result<binary::Binary<'b>, BadArgument> {
        let available_byte_count = Term::heap_binary_to_byte_count(&self.header);
        let PartRange {
            byte_offset,
            byte_count,
        } = start_length_to_part_range(start, length, available_byte_count)?;

        if (byte_offset == 0) & (byte_count == available_byte_count) {
            Ok(binary::Binary::Heap(self))
        } else {
            let process_subbinary = process.subbinary(self.into(), byte_offset, 0, byte_count, 0);

            Ok(binary::Binary::Sub(process_subbinary))
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
        let available_byte_count = Term::heap_binary_to_byte_count(&self.header);
        let part_range = start_length_to_part_range(start, length, available_byte_count)?;
        let list = part_range_to_list(self.iter(), part_range, &mut process);

        Ok(list)
    }
}

impl OrderInProcess for Binary {
    fn cmp_in_process(&self, other: &Binary, process: &Process) -> Ordering {
        match self.header.cmp_in_process(&other.header, process) {
            Ordering::Equal => {
                let mut final_ordering = Ordering::Equal;

                for (self_element, other_element) in self.iter().zip(other.iter()) {
                    match self_element.cmp(&other_element) {
                        Ordering::Equal => continue,
                        ordering => {
                            final_ordering = ordering;
                            break;
                        }
                    }
                }

                final_ordering
            }
            ordering => ordering,
        }
    }
}

impl From<&Binary> for Vec<u8> {
    fn from(binary: &Binary) -> Vec<u8> {
        let mut bytes_vec: Vec<u8> = Vec::with_capacity(binary.byte_size());
        bytes_vec.extend(binary.byte_iter());

        bytes_vec
    }
}

impl ToTerm for Binary {
    fn to_term(
        &self,
        options: ToTermOptions,
        mut process: &mut Process,
    ) -> Result<Term, BadArgument> {
        let mut iter = self.iter();

        match iter.next_versioned_term(options.existence, &mut process) {
            Some(term) => {
                if options.used {
                    let used = self.byte_size() - iter.len();
                    let used_term: Term = used.into_process(&mut process);

                    Ok(Term::slice_to_tuple(&[term, used_term], &mut process))
                } else {
                    Ok(term)
                }
            }
            None => Err(bad_argument!()),
        }
    }
}

impl TryFrom<&Binary> for String {
    type Error = BadArgument;

    fn try_from(binary: &Binary) -> Result<String, BadArgument> {
        let byte_vec: Vec<u8> = binary.into();

        String::from_utf8(byte_vec).map_err(|_| bad_argument!())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod from_slice {
        use super::*;

        #[test]
        fn without_bytes() {
            let mut byte_arena: TypedArena<u8> = Default::default();
            let mut binary_arena: TypedArena<Binary> = Default::default();

            let binary = Binary::from_slice(&[], &mut binary_arena, &mut byte_arena);

            assert_eq!(binary.header.tagged, Term::heap_binary(0).tagged);
        }

        #[test]
        fn with_bytes() {
            let mut byte_arena: TypedArena<u8> = Default::default();
            let mut binary_arena: TypedArena<Binary> = Default::default();

            let binary = Binary::from_slice(&[0, 1, 2, 3], &mut binary_arena, &mut byte_arena);

            assert_eq!(binary.header.tagged, Term::heap_binary(4).tagged);
            assert_eq!(unsafe { *binary.bytes.offset(0) }, 0);
            assert_eq!(unsafe { *binary.bytes.offset(1) }, 1);
            assert_eq!(unsafe { *binary.bytes.offset(2) }, 2);
            assert_eq!(unsafe { *binary.bytes.offset(3) }, 3);
        }
    }

    mod eq {
        use super::*;

        #[test]
        fn without_elements() {
            let mut process: Process = Default::default();
            let binary =
                Binary::from_slice(&[], &mut process.heap_binary_arena, &mut process.byte_arena);
            let equal =
                Binary::from_slice(&[], &mut process.heap_binary_arena, &mut process.byte_arena);

            assert_eq_in_process!(binary, binary, process);
            assert_eq_in_process!(binary, equal, process);
        }

        #[test]
        fn without_equal_length() {
            let mut process: Process = Default::default();
            let binary = Binary::from_slice(
                &[0],
                &mut process.heap_binary_arena,
                &mut process.byte_arena,
            );
            let unequal = Binary::from_slice(
                &[0, 1],
                &mut process.heap_binary_arena,
                &mut process.byte_arena,
            );

            assert_ne_in_process!(binary, unequal, process);
        }

        #[test]
        fn with_equal_length_without_same_byte() {
            let mut process: Process = Default::default();
            let binary = Binary::from_slice(
                &[0],
                &mut process.heap_binary_arena,
                &mut process.byte_arena,
            );
            let unequal = Binary::from_slice(
                &[1],
                &mut process.heap_binary_arena,
                &mut process.byte_arena,
            );

            assert_eq_in_process!(binary, binary, process);
            assert_ne_in_process!(binary, unequal, process);
        }

        #[test]
        fn with_equal_length_with_same_bytes() {
            let mut process: Process = Default::default();
            let binary = Binary::from_slice(
                &[0],
                &mut process.heap_binary_arena,
                &mut process.byte_arena,
            );
            let unequal = Binary::from_slice(
                &[0],
                &mut process.heap_binary_arena,
                &mut process.byte_arena,
            );

            assert_eq_in_process!(binary, unequal, process);
        }
    }

    mod iter {
        use super::*;

        use std::convert::TryInto;

        #[test]
        fn without_elements() {
            let mut process: Process = Default::default();
            let binary =
                Binary::from_slice(&[], &mut process.heap_binary_arena, &mut process.byte_arena);

            assert_eq!(binary.iter().count(), 0);

            let size_integer: Integer = binary.size();
            let size_usize: usize = size_integer.try_into().unwrap();

            assert_eq!(binary.iter().count(), size_usize);
        }

        #[test]
        fn with_elements() {
            let mut process: Process = Default::default();
            let binary = Binary::from_slice(
                &[0],
                &mut process.heap_binary_arena,
                &mut process.byte_arena,
            );

            assert_eq!(binary.iter().count(), 1);

            let size_integer: Integer = binary.size();
            let size_usize: usize = size_integer.try_into().unwrap();

            assert_eq!(binary.iter().count(), size_usize);
        }

        #[test]
        fn is_double_ended() {
            let mut process: Process = Default::default();
            let binary = Binary::from_slice(
                &[0, 1, 2],
                &mut process.heap_binary_arena,
                &mut process.byte_arena,
            );

            let mut iter = binary.iter();

            assert_eq!(iter.next(), Some(0));
            assert_eq!(iter.next(), Some(1));
            assert_eq!(iter.next(), Some(2));
            assert_eq!(iter.next(), None);
            assert_eq!(iter.next(), None);

            let mut rev_iter = binary.iter();

            assert_eq!(rev_iter.next_back(), Some(2));
            assert_eq!(rev_iter.next_back(), Some(1));
            assert_eq!(rev_iter.next_back(), Some(0));
            assert_eq!(rev_iter.next_back(), None);
            assert_eq!(rev_iter.next_back(), None);

            let mut double_ended_iter = binary.iter();

            assert_eq!(double_ended_iter.next(), Some(0));
            assert_eq!(double_ended_iter.next_back(), Some(2));
            assert_eq!(double_ended_iter.next(), Some(1));
            assert_eq!(double_ended_iter.next_back(), None);
            assert_eq!(double_ended_iter.next(), None);
        }
    }

    mod size {
        use super::*;

        #[test]
        fn without_elements() {
            let mut process: Process = Default::default();
            let binary =
                Binary::from_slice(&[], &mut process.heap_binary_arena, &mut process.byte_arena);

            assert_eq_in_process!(binary.size(), &0.into(), process);
        }

        #[test]
        fn with_elements() {
            let mut process: Process = Default::default();
            let binary = Binary::from_slice(
                &[0],
                &mut process.heap_binary_arena,
                &mut process.byte_arena,
            );

            assert_eq_in_process!(binary.size(), &1.into(), process);
        }
    }
}
