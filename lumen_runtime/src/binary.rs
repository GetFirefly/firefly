use std::convert::TryInto;
use std::mem::transmute;

use crate::atom::Existence;
use crate::list::ToList;
use crate::process::{IntoProcess, Process};
use crate::term::{self, BadArgument, Term};

pub mod heap;
pub mod sub;

pub enum Binary<'a> {
    Heap(&'a heap::Binary),
    Sub(&'a sub::Binary),
}

impl<'a> Binary<'a> {
    pub fn from_slice(bytes: &[u8], process: &mut Process) -> Self {
        // TODO use reference counted binaries for bytes.len() > 64
        let heap_binary = heap::Binary::from_slice(
            bytes,
            &mut process.heap_binary_arena,
            &mut process.byte_arena,
        );

        Binary::Heap(heap_binary)
    }
}

trait ByteIterator: ExactSizeIterator + DoubleEndedIterator + Iterator<Item = u8>
where
    Self: Sized,
{
    fn next_f64(&mut self) -> Option<f64> {
        match (
            self.next(),
            self.next(),
            self.next(),
            self.next(),
            self.next(),
            self.next(),
            self.next(),
            self.next(),
        ) {
            (
                Some(first_byte),
                Some(second_byte),
                Some(third_byte),
                Some(fourth_byte),
                Some(fifth_byte),
                Some(sixth_byte),
                Some(seventh_byte),
                Some(eighth_byte),
            ) => {
                let unsigned = ((first_byte as u64) << 56)
                    | ((second_byte as u64) << 48)
                    | ((third_byte as u64) << 40)
                    | ((fourth_byte as u64) << 32)
                    | ((fifth_byte as u64) << 24)
                    | ((sixth_byte as u64) << 16)
                    | ((seventh_byte as u64) << 8)
                    | (eighth_byte as u64);

                Some(unsafe { transmute(unsigned) })
            }
            _ => None,
        }
    }

    fn next_i32(&mut self) -> Option<i32> {
        match (self.next(), self.next(), self.next(), self.next()) {
            (Some(first_byte), Some(second_byte), Some(third_byte), Some(fourth_byte)) => Some(
                ((first_byte as i32) << 24)
                    | ((second_byte as i32) << 16)
                    | ((third_byte as i32) << 8)
                    | (fourth_byte as i32),
            ),
            _ => None,
        }
    }

    fn next_u16(&mut self) -> Option<u16> {
        match (self.next(), self.next()) {
            (Some(first_byte), Some(second_byte)) => {
                Some(((first_byte as u16) << 8) | (second_byte as u16))
            }
            _ => None,
        }
    }

    fn next_u32(&mut self) -> Option<u32> {
        match (self.next(), self.next(), self.next(), self.next()) {
            (Some(first_byte), Some(second_byte), Some(third_byte), Some(fourth_byte)) => Some(
                ((first_byte as u32) << 24)
                    | ((second_byte as u32) << 16)
                    | ((third_byte as u32) << 8)
                    | (fourth_byte as u32),
            ),
            _ => None,
        }
    }

    fn next_atom(&mut self, mut process: &mut Process) -> Option<Term> {
        self.next_u16()
            .and_then(|length| self.next_byte_vec(length as usize))
            .and_then(|byte_vec| match String::from_utf8(byte_vec) {
                Ok(string) => {
                    match Term::str_to_atom(&string, Existence::DoNotCare, &mut process) {
                        Ok(term) => Some(term),
                        Err(BadArgument) => None,
                    }
                }
                Err(_) => None,
            })
    }

    fn next_binary(&mut self, mut process: &mut Process) -> Option<Term> {
        self.next_u32()
            .and_then(|length| self.next_byte_vec(length as usize))
            .map(|byte_vec| Term::slice_to_binary(byte_vec.as_slice(), &mut process))
    }

    fn next_bit_binary(&mut self, mut process: &mut Process) -> Option<Term> {
        self.next_u32().and_then(|binary_byte_count| {
            self.next().and_then(|bit_count| {
                self.next_byte_vec(binary_byte_count as usize)
                    .map(|byte_vec| {
                        let original = Term::slice_to_binary(byte_vec.as_slice(), &mut process);

                        Term::subbinary(
                            original,
                            0,
                            0,
                            (binary_byte_count - 1) as usize,
                            bit_count,
                            &mut process,
                        )
                    })
            })
        })
    }

    fn next_byte_list(&mut self, mut process: &mut Process) -> Option<Term> {
        self.next_u16()
            .and_then(|length| self.next_byte_vec(length as usize))
            .map(|byte_vec| {
                byte_vec.iter().rfold(Term::EMPTY_LIST, |acc, element| {
                    Term::cons(element.into_process(&mut process), acc, &mut process)
                })
            })
    }

    fn next_byte_vec(&mut self, length: usize) -> Option<Vec<u8>> {
        let mut byte_vec: Vec<u8> = Vec::with_capacity(length);

        for _ in 0..length {
            match self.next() {
                Some(byte) => byte_vec.push(byte),
                None => break,
            }
        }

        if byte_vec.len() == length {
            Some(byte_vec)
        } else {
            None
        }
    }

    fn next_small_integer(&mut self) -> Option<Term> {
        self.next().map(|byte| byte.into())
    }

    fn next_integer(&mut self, mut process: &mut Process) -> Option<Term> {
        self.next_i32()
            .map(|integer| integer.into_process(&mut process))
    }

    fn next_list(&mut self, mut process: &mut Process) -> Option<Term> {
        self.next_u32()
            .and_then(|element_count| self.next_terms(element_count as usize, &mut process))
            .and_then(|element_vec| {
                self.next_term(&mut process).map(|tail_term| {
                    element_vec.iter().rfold(tail_term, |acc, element| {
                        Term::cons(*element, acc, &mut process)
                    })
                })
            })
    }

    fn next_new_float(&mut self, mut process: &mut Process) -> Option<Term> {
        self.next_f64()
            .map(|inner| inner.into_process(&mut process))
    }

    fn next_small_atom_utf8(&mut self, mut process: &mut Process) -> Option<Term> {
        self.next()
            .and_then(|length| self.next_byte_vec(length as usize))
            .and_then(|byte_vec| match String::from_utf8(byte_vec) {
                Ok(string) => {
                    match Term::str_to_atom(&string, Existence::DoNotCare, &mut process) {
                        Ok(term) => Some(term),
                        Err(BadArgument) => None,
                    }
                }
                Err(_) => None,
            })
    }

    fn next_small_big_integer(&mut self, mut process: &mut Process) -> Option<Term> {
        self.next().and_then(|count| {
            self.next().and_then(|sign| {
                let mut rug_integer = rug::Integer::new();
                let mut truncated = false;

                for _ in 0..count {
                    match self.next() {
                        Some(byte) => rug_integer = (rug_integer << 8) | (byte as u32),
                        None => {
                            truncated = true;
                            break;
                        }
                    }
                }

                if truncated {
                    None
                } else {
                    let signed_rug_integer = if sign == 0 {
                        rug_integer
                    } else {
                        -1 * rug_integer
                    };

                    Some(signed_rug_integer.into_process(&mut process))
                }
            })
        })
    }

    fn next_small_tuple(&mut self, mut process: &mut Process) -> Option<Term> {
        self.next()
            .and_then(|arity| self.next_terms(arity as usize, &mut process))
            .map(|element_vec| Term::slice_to_tuple(element_vec.as_slice(), &mut process))
    }

    fn next_term(&mut self, mut process: &mut Process) -> Option<Term> {
        match self.next() {
            Some(tag_byte) => match tag_byte.try_into() {
                Ok(tag) => {
                    use crate::term::external_format::Tag::*;

                    match tag {
                        Atom => self.next_atom(&mut process),
                        Binary => self.next_binary(&mut process),
                        BitBinary => self.next_bit_binary(&mut process),
                        ByteList => self.next_byte_list(&mut process),
                        EmptyList => Some(Term::EMPTY_LIST),
                        Integer => self.next_integer(&mut process),
                        List => self.next_list(&mut process),
                        NewFloat => self.next_new_float(&mut process),
                        SmallAtomUTF8 => self.next_small_atom_utf8(&mut process),
                        SmallBigInteger => self.next_small_big_integer(&mut process),
                        SmallInteger => self.next_small_integer(),
                        SmallTuple => self.next_small_tuple(&mut process),
                    }
                }
                _ => None,
            },
            None => None,
        }
    }

    fn next_terms(&mut self, count: usize, mut process: &mut Process) -> Option<Vec<Term>> {
        let mut element_vec: Vec<Term> = Vec::with_capacity(count);

        for _ in 0..count {
            match self.next_term(&mut process) {
                Some(term) => element_vec.push(term),
                None => break,
            }
        }

        if element_vec.len() == count {
            Some(element_vec)
        } else {
            None
        }
    }

    fn next_versioned_term(&mut self, mut process: &mut Process) -> Option<Term> {
        match self.next() {
            Some(term::external_format::VERSION_NUMBER) => self.next_term(&mut process),
            Some(version) => panic!("Unknown version number ({})", version),
            None => None,
        }
    }

    fn part_range(
        &mut self,
        PartRange {
            byte_offset,
            byte_count,
        }: PartRange,
    ) -> &mut Self {
        // skip byte_offset
        for _ in 0..byte_offset {
            self.next();
        }

        for _ in byte_count..self.len() {
            self.next_back();
        }

        self
    }
}

pub trait Part<'a, S, L, T> {
    fn part(&'a self, start: S, length: L, process: &mut Process) -> Result<T, BadArgument>;
}

pub struct PartRange {
    byte_offset: usize,
    byte_count: usize,
}

fn start_length_to_part_range(
    start: usize,
    length: isize,
    available_byte_count: usize,
) -> Result<PartRange, BadArgument> {
    if length >= 0 {
        let non_negative_length = length as usize;

        if (start < available_byte_count) & (start + non_negative_length <= available_byte_count) {
            Ok(PartRange {
                byte_offset: start,
                byte_count: non_negative_length,
            })
        } else {
            Err(BadArgument)
        }
    } else {
        let start_isize = start as isize;

        if (start <= available_byte_count) & (0 <= start_isize + length) {
            let byte_offset = (start_isize + length) as usize;
            let byte_count = (-length) as usize;

            Ok(PartRange {
                byte_offset,
                byte_count,
            })
        } else {
            Err(BadArgument)
        }
    }
}

fn part_range_to_list<T: ByteIterator>(
    mut byte_iterator: T,
    part_range: PartRange,
    mut process: &mut Process,
) -> Term {
    byte_iterator.part_range(part_range).to_list(&mut process)
}

pub trait PartToList<S, L> {
    fn part_to_list(&self, start: S, length: L, process: &mut Process)
        -> Result<Term, BadArgument>;
}

pub trait ToTerm {
    fn to_term(&self, process: &mut Process) -> Result<Term, BadArgument>;
}
