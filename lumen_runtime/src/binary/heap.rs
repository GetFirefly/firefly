use std::cmp::Ordering::{self, *};
use std::convert::TryFrom;
use std::fmt::{self, Debug};
use std::hash::{Hash, Hasher};

use crate::atom::{self, Existence};
use crate::binary::{
    self, part_range_to_list, start_length_to_part_range, sub, ByteIterator, Part, PartRange,
    PartToList, ToTerm, ToTermOptions,
};
use crate::exception::Exception;
use crate::integer::Integer;
use crate::process::{IntoProcess, Process};
use crate::term::Term;

pub struct Binary {
    header: Term,
    bytes: *const u8,
}

impl Binary {
    pub fn from_slice(bytes: &[u8], process: &mut Process) -> &'static Self {
        let arena_bytes: &[u8] = if bytes.len() != 0 {
            process.byte_arena.alloc_slice(bytes)
        } else {
            &[]
        };

        let pointer = process.heap_binary_arena.alloc(Binary::new(arena_bytes)) as *const Binary;

        unsafe { &*pointer }
    }

    fn new(bytes: &[u8]) -> Self {
        Binary {
            header: Term::heap_binary(bytes.len()),
            bytes: bytes.as_ptr(),
        }
    }

    pub fn as_slice(&self) -> &'static [u8] {
        unsafe { std::slice::from_raw_parts(self.bytes, self.header.heap_binary_to_byte_count()) }
    }

    pub fn bit_len(&self) -> usize {
        self.byte_len() * 8
    }

    pub fn byte(&self, index: usize) -> u8 {
        let byte_count = self.byte_len();

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

    pub fn byte_len(&self) -> usize {
        unsafe { self.header.heap_binary_to_byte_count() }
    }

    pub fn iter(&self) -> Iter {
        let byte_count = self.byte_len();

        unsafe {
            Iter {
                pointer: self.bytes,
                limit: self.bytes.offset(byte_count as isize),
            }
        }
    }

    pub fn len(&self) -> usize {
        self.byte_len()
    }

    pub fn size(&self) -> Integer {
        // The `header` field is not the same as `size` because `size` is tagged as a small integer
        // while `header` is tagged as `HeapBinary` to mark the beginning of a heap binary.
        self.len().into()
    }

    pub fn to_atom_index(&self, existence: Existence) -> Option<atom::Index> {
        let bytes = self.as_slice();

        atom::str_to_index(std::str::from_utf8(bytes).unwrap(), existence)
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

impl Debug for Binary {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Binary::from_slice(&[")?;

        let mut iter = self.iter();

        if let Some(first_byte) = iter.next() {
            write!(f, "{:?}", first_byte)?;

            for byte in iter {
                write!(f, ", {:?}", byte)?;
            }
        }

        write!(f, "])")
    }
}

impl Eq for Binary {}

impl Hash for Binary {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for byte in self.byte_iter() {
            byte.hash(state);
        }
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
    ) -> Result<binary::Binary<'b>, Exception> {
        let available_byte_count = self.byte_len();
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

impl PartialEq for Binary {
    fn eq(&self, other: &Binary) -> bool {
        match self.header.tagged == other.header.tagged {
            true => {
                let mut final_eq = true;

                for (self_element, other_element) in self.iter().zip(other.iter()) {
                    match self_element == other_element {
                        true => continue,
                        eq => {
                            final_eq = eq;
                            break;
                        }
                    }
                }

                final_eq
            }
            eq => eq,
        }
    }
}

impl PartialEq<sub::Binary> for Binary {
    /// > * Bitstrings are compared byte by byte, incomplete bytes are compared bit by bit.
    /// > -- https://hexdocs.pm/elixir/operators.html#term-ordering
    fn eq(&self, other: &sub::Binary) -> bool {
        (other.bit_count == 0) & self.byte_iter().eq(other.byte_iter())
    }
}

impl PartialOrd for Binary {
    fn partial_cmp(&self, other: &Binary) -> Option<Ordering> {
        match self.len().partial_cmp(&other.len()) {
            Some(Equal) => self.iter().partial_cmp(other.iter()),
            partial_ordering => partial_ordering,
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
        let available_byte_count = self.byte_len();
        let part_range = start_length_to_part_range(start, length, available_byte_count)?;
        let list = part_range_to_list(self.iter(), part_range, &mut process);

        Ok(list)
    }
}

impl From<&Binary> for Vec<u8> {
    fn from(binary: &Binary) -> Vec<u8> {
        let mut bytes_vec: Vec<u8> = Vec::with_capacity(binary.byte_len());
        bytes_vec.extend(binary.byte_iter());

        bytes_vec
    }
}

impl ToTerm for Binary {
    fn to_term(
        &self,
        options: ToTermOptions,
        mut process: &mut Process,
    ) -> Result<Term, Exception> {
        let mut iter = self.iter();

        match iter.next_versioned_term(options.existence, &mut process) {
            Some(term) => {
                if options.used {
                    let used = self.byte_len() - iter.len();
                    let used_term: Term = used.into_process(&mut process);

                    Ok(Term::slice_to_tuple(&[term, used_term], &mut process))
                } else {
                    Ok(term)
                }
            }
            None => Err(badarg!()),
        }
    }
}

impl TryFrom<&Binary> for String {
    type Error = Exception;

    fn try_from(binary: &Binary) -> Result<String, Exception> {
        let byte_vec: Vec<u8> = binary.into();

        String::from_utf8(byte_vec).map_err(|_| badarg!())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod from_slice {
        use super::*;

        use std::sync::{Arc, RwLock};

        use crate::environment::{self, Environment};

        #[test]
        fn without_bytes() {
            let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
            let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
            let mut process = process_rw_lock.write().unwrap();

            let binary = Binary::from_slice(&[], &mut process);

            assert_eq!(binary.header.tagged, Term::heap_binary(0).tagged);
        }

        #[test]
        fn with_bytes() {
            let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
            let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
            let mut process = process_rw_lock.write().unwrap();

            let binary = Binary::from_slice(&[0, 1, 2, 3], &mut process);

            assert_eq!(binary.header.tagged, Term::heap_binary(4).tagged);
            assert_eq!(unsafe { *binary.bytes.offset(0) }, 0);
            assert_eq!(unsafe { *binary.bytes.offset(1) }, 1);
            assert_eq!(unsafe { *binary.bytes.offset(2) }, 2);
            assert_eq!(unsafe { *binary.bytes.offset(3) }, 3);
        }
    }

    mod eq {
        use super::*;

        use std::sync::{Arc, RwLock};

        use crate::environment::{self, Environment};

        #[test]
        fn without_elements() {
            let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
            let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
            let mut process = process_rw_lock.write().unwrap();
            let binary = Binary::from_slice(&[], &mut process);
            let equal = Binary::from_slice(&[], &mut process);

            assert_eq!(binary, binary);
            assert_eq!(binary, equal);
        }

        #[test]
        fn without_equal_length() {
            let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
            let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
            let mut process = process_rw_lock.write().unwrap();
            let binary = Binary::from_slice(&[0], &mut process);
            let unequal = Binary::from_slice(&[0, 1], &mut process);

            assert_ne!(binary, unequal);
        }

        #[test]
        fn with_equal_length_without_same_byte() {
            let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
            let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
            let mut process = process_rw_lock.write().unwrap();
            let binary = Binary::from_slice(&[0], &mut process);
            let unequal = Binary::from_slice(&[1], &mut process);

            assert_eq!(binary, binary);
            assert_ne!(binary, unequal);
        }

        #[test]
        fn with_equal_length_with_same_bytes() {
            let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
            let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
            let mut process = process_rw_lock.write().unwrap();
            let binary = Binary::from_slice(&[0], &mut process);
            let unequal = Binary::from_slice(&[0], &mut process);

            assert_eq!(binary, unequal);
        }
    }

    mod iter {
        use super::*;

        use std::convert::TryInto;
        use std::sync::{Arc, RwLock};

        use crate::environment::{self, Environment};

        #[test]
        fn without_elements() {
            let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
            let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
            let mut process = process_rw_lock.write().unwrap();
            let binary = Binary::from_slice(&[], &mut process);

            assert_eq!(binary.iter().count(), 0);

            let size_integer: Integer = binary.size();
            let size_usize: usize = size_integer.try_into().unwrap();

            assert_eq!(binary.iter().count(), size_usize);
        }

        #[test]
        fn with_elements() {
            let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
            let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
            let mut process = process_rw_lock.write().unwrap();
            let binary = Binary::from_slice(&[0], &mut process);

            assert_eq!(binary.iter().count(), 1);

            let size_integer: Integer = binary.size();
            let size_usize: usize = size_integer.try_into().unwrap();

            assert_eq!(binary.iter().count(), size_usize);
        }

        #[test]
        fn is_double_ended() {
            let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
            let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
            let mut process = process_rw_lock.write().unwrap();
            let binary = Binary::from_slice(&[0, 1, 2], &mut process);

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

        use std::sync::{Arc, RwLock};

        use crate::environment::{self, Environment};

        #[test]
        fn without_elements() {
            let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
            let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
            let mut process = process_rw_lock.write().unwrap();
            let binary = Binary::from_slice(&[], &mut process);

            assert_eq!(binary.size(), 0.into());
        }

        #[test]
        fn with_elements() {
            let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
            let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
            let mut process = process_rw_lock.write().unwrap();
            let binary = Binary::from_slice(&[0], &mut process);

            assert_eq!(binary.size(), 1.into());
        }
    }
}
