use std::sync::Arc;

use proptest::strategy::{BoxedStrategy, Just, Strategy};

use liblumen_alloc::erts::term::Term;
use liblumen_alloc::erts::ProcessControlBlock;

use crate::otp::erlang::tests::strategy::{self, bits_to_bytes, size_range};

pub mod byte_count;
pub mod is_binary;

pub fn bit_count() -> BoxedStrategy<u8> {
    bit()
}

pub fn bit_offset() -> BoxedStrategy<u8> {
    bit()
}

pub fn byte_count() -> BoxedStrategy<usize> {
    size_range::strategy()
}

pub fn byte_offset() -> BoxedStrategy<usize> {
    size_range::strategy()
}

pub fn containing_bytes(
    byte_vec: Vec<u8>,
    arc_process: Arc<ProcessControlBlock>,
) -> BoxedStrategy<Term> {
    (byte_offset(), bit_offset(), Just(byte_vec))
        .prop_flat_map(|(byte_offset, bit_offset, byte_vec)| {
            let original_bit_len = original_bit_len(byte_offset, bit_offset, byte_vec.len(), 0);
            let original_byte_len = bits_to_bytes(original_bit_len);

            (
                Just(byte_offset),
                Just(bit_offset),
                Just(byte_vec),
                strategy::byte_vec::with_size_range((original_byte_len..=original_byte_len).into()),
            )
        })
        .prop_map(
            move |(byte_offset, bit_offset, byte_vec, mut original_byte_vec)| {
                write_bytes(&mut original_byte_vec, byte_offset, bit_offset, &byte_vec);

                let original = arc_process.binary_from_bytes(&original_byte_vec).unwrap();

                arc_process
                    .subbinary_from_original(original, byte_offset, bit_offset, byte_vec.len(), 0)
                    .unwrap()
            },
        )
        .boxed()
}

pub fn is_binary(arc_process: Arc<ProcessControlBlock>) -> BoxedStrategy<Term> {
    with_size_range(
        byte_offset(),
        bit_offset(),
        byte_count(),
        (0_u8..=0_u8).boxed(),
        arc_process,
    )
}

pub fn is_not_binary(arc_process: Arc<ProcessControlBlock>) -> BoxedStrategy<Term> {
    with_size_range(
        byte_offset(),
        bit_offset(),
        byte_count(),
        (1_u8..=7_u8).boxed(),
        arc_process,
    )
}

fn original_bit_len(byte_offset: usize, bit_offset: u8, byte_count: usize, bit_count: u8) -> usize {
    byte_offset * 8 + bit_offset as usize + byte_count * 8 + bit_count as usize
}

pub fn with_bit_count(bit_count: u8, arc_process: Arc<ProcessControlBlock>) -> BoxedStrategy<Term> {
    with_size_range(
        byte_offset(),
        bit_offset(),
        byte_count(),
        (bit_count..=bit_count).boxed(),
        arc_process,
    )
}

pub fn with_size_range(
    byte_offset: BoxedStrategy<usize>,
    bit_offset: BoxedStrategy<u8>,
    byte_count: BoxedStrategy<usize>,
    bit_count: BoxedStrategy<u8>,
    arc_process: Arc<ProcessControlBlock>,
) -> BoxedStrategy<Term> {
    let original_arc_process = arc_process.clone();
    let subbinary_arc_process = arc_process.clone();

    (byte_offset, bit_offset, byte_count, bit_count)
        .prop_flat_map(move |(byte_offset, bit_offset, byte_count, bit_count)| {
            let original_bit_len = original_bit_len(byte_offset, bit_offset, byte_count, bit_count);
            let original_byte_len = bits_to_bytes(original_bit_len);

            let original = super::heap::with_size_range(
                (original_byte_len..=original_byte_len).into(),
                original_arc_process.clone(),
            );

            (
                Just(byte_offset),
                Just(bit_offset),
                Just(byte_count),
                Just(bit_count),
                original,
            )
        })
        .prop_map(
            move |(byte_offset, bit_offset, byte_count, bit_count, original)| {
                subbinary_arc_process
                    .subbinary_from_original(
                        original,
                        byte_offset,
                        bit_offset,
                        byte_count,
                        bit_count,
                    )
                    .unwrap()
            },
        )
        .boxed()
}

// Private

fn bit() -> BoxedStrategy<u8> {
    (0_u8..=7_u8).boxed()
}

fn write_bytes(original_byte_vec: &mut [u8], byte_offset: usize, bit_offset: u8, bytes: &[u8]) {
    for (i, byte) in bytes.iter().enumerate() {
        let first_original_byte_index = byte_offset + i;
        let first_original_byte = original_byte_vec[first_original_byte_index];
        let first_byte = byte >> bit_offset;
        let first_byte_mask: u8 = 0xFF >> bit_offset;

        original_byte_vec[first_original_byte_index] =
            (first_original_byte & !first_byte_mask) | first_byte;

        if bit_offset > 0 {
            let second_original_byte_index = first_original_byte_index + 1;
            let second_original_byte = original_byte_vec[second_original_byte_index];
            let second_byte_shift = 8 - bit_offset;
            let second_byte = byte << second_byte_shift;
            let second_byte_mask: u8 = 0xFF << second_byte_shift;

            original_byte_vec[second_original_byte_index] =
                (second_original_byte & !second_byte_mask) | second_byte;
        }
    }
}
