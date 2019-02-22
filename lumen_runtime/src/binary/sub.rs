use std::cmp::Ordering;
use std::iter::FusedIterator;

use crate::binary::heap;
use crate::process::{OrderInProcess, Process};
use crate::term::{Tag, Term};

pub struct Binary {
    header: Term,
    original: Term,
    byte_offset: usize,
    bit_offset: u8,
    byte_count: usize,
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

    pub fn byte_iter(&self) -> ByteIter {
        ByteIter {
            original: self.original,
            byte_offset: self.byte_offset,
            bit_offset: self.bit_offset,
            current_byte_count: 0,
            max_byte_count: self.byte_count,
        }
    }

    pub fn last_bits_byte(&self) -> u8 {
        if 0 < self.bit_offset {
            self.original.byte(self.byte_count + 1) >> (8 - self.bit_offset)
        } else {
            0
        }
    }
}

pub struct ByteIter {
    original: Term,
    byte_offset: usize,
    bit_offset: u8,
    current_byte_count: usize,
    max_byte_count: usize,
}

impl Iterator for ByteIter {
    type Item = u8;

    fn next(&mut self) -> Option<u8> {
        if self.current_byte_count == self.max_byte_count {
            None
        } else {
            let first_index = self.byte_offset + self.current_byte_count;
            let first_byte = self.original.byte(first_index);
            self.current_byte_count += 1;

            if 0 < self.bit_offset {
                let second_byte = self.original.byte(first_index + 1);

                Some((first_byte << self.bit_offset) | (second_byte >> (8 - self.bit_offset)))
            } else {
                Some(first_byte)
            }
        }
    }
}

impl FusedIterator for ByteIter {}

impl OrderInProcess<heap::Binary> for Binary {
    /// > * Bitstrings are compared byte by byte, incomplete bytes are compared bit by bit.
    /// > -- https://hexdocs.pm/elixir/operators.html#term-ordering
    fn cmp_in_process(&self, other: &heap::Binary, _process: &Process) -> Ordering {
        let mut final_ordering = Ordering::Equal;
        let mut self_byte_iter = self.byte_iter();
        let mut other_byte_iter = other.byte_iter();

        while final_ordering == Ordering::Equal {
            final_ordering = match (self_byte_iter.next(), other_byte_iter.next()) {
                (Some(ref self_byte), Some(ref other_byte)) => self_byte.cmp(other_byte),
                (Some(_), None) => Ordering::Greater,
                (None, Some(_)) => Ordering::Less,
                (None, None) => break,
            }
        }

        if final_ordering == Ordering::Equal {
            // a heap::Binary has no tail bits, so if the subbinary has any tail bits it is greater
            if self.bit_count > 0 {
                final_ordering = Ordering::Greater
            }
        }

        final_ordering
    }
}
