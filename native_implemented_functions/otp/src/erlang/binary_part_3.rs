// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;

use anyhow::*;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use native_implemented_function::native_implemented_function;

use crate::binary::{start_length_to_part_range, PartRange};
use crate::runtime::context::*;

#[native_implemented_function(binary_part/3)]
pub fn result(
    process: &Process,
    binary: Term,
    start: Term,
    length: Term,
) -> exception::Result<Term> {
    let start_usize: usize = start
        .try_into()
        .with_context(|| term_is_not_non_negative_integer("start", start))?;
    let length_isize = term_try_into_isize!(length)?;

    match binary.decode().unwrap() {
        TypedTerm::HeapBinary(heap_binary) => {
            let available_byte_count = heap_binary.full_byte_len();
            let PartRange {
                byte_offset,
                byte_len,
            } = start_length_to_part_range(start_usize, length_isize, available_byte_count)?;

            if (byte_offset == 0) && (byte_len == available_byte_count) {
                Ok(binary)
            } else {
                process
                    .subbinary_from_original(binary, byte_offset, 0, byte_len, 0)
                    .map_err(|error| error.into())
            }
        }
        TypedTerm::ProcBin(process_binary) => {
            let available_byte_count = process_binary.full_byte_len();
            let PartRange {
                byte_offset,
                byte_len,
            } = start_length_to_part_range(start_usize, length_isize, available_byte_count)?;

            if (byte_offset == 0) && (byte_len == available_byte_count) {
                Ok(binary)
            } else {
                process
                    .subbinary_from_original(binary, byte_offset, 0, byte_len, 0)
                    .map_err(|error| error.into())
            }
        }
        TypedTerm::SubBinary(subbinary) => {
            let PartRange {
                byte_offset,
                byte_len,
            } = start_length_to_part_range(start_usize, length_isize, subbinary.full_byte_len())?;

            // new subbinary is entire subbinary
            if (subbinary.is_binary())
                && (byte_offset == 0)
                && (byte_len == subbinary.full_byte_len())
            {
                Ok(binary)
            } else {
                process
                    .subbinary_from_original(
                        subbinary.original(),
                        subbinary.byte_offset() + byte_offset,
                        subbinary.bit_offset(),
                        byte_len,
                        0,
                    )
                    .map_err(|error| error.into())
            }
        }
        _ => Err(TypeError)
            .context(format!(
                "binary ({}) must be a binary or bitstring with at least 1 full byte",
                binary
            ))
            .map_err(From::from),
    }
}
