use std::sync::Arc;

use proptest::strategy::{Just, Strategy};

use crate::process::Process;
use crate::term::Term;

use super::super::super::{bits_to_bytes, size_range};

pub mod byte_count;
pub mod is_binary;

pub fn bit_count() -> impl Strategy<Value = u8> {
    bit()
}

pub fn bit_offset() -> impl Strategy<Value = u8> {
    bit()
}

pub fn byte_count() -> impl Strategy<Value = usize> {
    size_range::strategy()
}

pub fn byte_offset() -> impl Strategy<Value = usize> {
    size_range::strategy()
}

pub fn is_binary(arc_process: Arc<Process>) -> impl Strategy<Value = Term> {
    with_size_range(
        byte_offset(),
        bit_offset(),
        byte_count(),
        0_u8..=0_u8,
        arc_process,
    )
}

pub fn with_size_range(
    byte_offset: impl Strategy<Value = usize>,
    bit_offset: impl Strategy<Value = u8>,
    byte_count: impl Strategy<Value = usize>,
    bit_count: impl Strategy<Value = u8>,
    arc_process: Arc<Process>,
) -> impl Strategy<Value = Term> {
    let original_arc_process = arc_process.clone();
    let subbinary_arc_process = arc_process.clone();

    (byte_offset, bit_offset, byte_count, bit_count)
        .prop_flat_map(move |(byte_offset, bit_offset, byte_count, bit_count)| {
            let original_bit_len =
                byte_offset * 8 + bit_offset as usize + byte_count * 8 + bit_count as usize;
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
                Term::subbinary(
                    original,
                    byte_offset,
                    bit_offset,
                    byte_count,
                    bit_count,
                    &subbinary_arc_process,
                )
            },
        )
}

// Private

fn bit() -> impl Strategy<Value = u8> {
    (0_u8..=7_u8).boxed()
}
