// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;

use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use lumen_runtime_macros::native_implemented_function;

use crate::binary::{start_length_to_part_range, PartRange};

#[native_implemented_function(binary_part/3)]
pub fn native(process: &Process, binary: Term, start: Term, length: Term) -> exception::Result {
    let start_usize: usize = start.try_into()?;
    let length_isize: isize = length.try_into()?;

    match binary.to_typed_term().unwrap() {
        TypedTerm::Boxed(unboxed_binary) => match unboxed_binary.to_typed_term().unwrap() {
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
                } = start_length_to_part_range(
                    start_usize,
                    length_isize,
                    subbinary.full_byte_len(),
                )?;

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
            _ => Err(badarg!().into()),
        },
        _ => Err(badarg!().into()),
    }
}
