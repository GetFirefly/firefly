// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;

use anyhow::*;

use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::alloc::TermAlloc;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use lumen_runtime_macros::native_implemented_function;

#[native_implemented_function(split_binary/2)]
pub fn native(process: &Process, binary: Term, position: Term) -> exception::Result<Term> {
    let index: usize = position
        .try_into()
        .context("positive must be in 0..byte_size(binary)")?;

    match binary.decode().unwrap() {
        binary_box @ TypedTerm::HeapBinary(_) | binary_box @ TypedTerm::ProcBin(_) => {
            if index == 0 {
                let mut heap = process.acquire_heap();

                let empty_prefix = heap
                    .subbinary_from_original(binary, index, 0, 0, 0)?
                    .encode()?;

                // Don't make a subbinary of the suffix since it is the same as the
                // `binary`.
                heap.tuple_from_slice(&[empty_prefix, binary])
                    .map_err(|e| e.into())
                    .and_then(|t| t.encode())
            } else {
                let full_byte_length = match binary_box {
                    TypedTerm::HeapBinary(heap_binary) => heap_binary.full_byte_len(),
                    TypedTerm::ProcBin(process_binary) => process_binary.full_byte_len(),
                    _ => unreachable!(),
                };

                if index < full_byte_length {
                    let mut heap = process.acquire_heap();
                    let prefix = heap
                        .subbinary_from_original(binary, 0, 0, index, 0)?
                        .encode()?;
                    let suffix = heap
                        .subbinary_from_original(binary, index, 0, full_byte_length - index, 0)?
                        .encode()?;

                    heap.tuple_from_slice(&[prefix, suffix])
                        .map_err(|e| e.into())
                        .and_then(|t| t.encode())
                } else if index == full_byte_length {
                    let mut heap = process.acquire_heap();
                    let empty_suffix = heap
                        .subbinary_from_original(binary, index, 0, 0, 0)?
                        .encode()?;

                    // Don't make a subbinary of the prefix since it is the same as the
                    // `binary`.
                    heap.tuple_from_slice(&[binary, empty_suffix])
                        .map_err(|e| e.into())
                        .and_then(|t| t.encode())
                } else {
                    Err(badarg!().into())
                }
            }
        }
        TypedTerm::SubBinary(subbinary) => {
            if index == 0 {
                let mut heap = process.acquire_heap();
                let empty_prefix = heap
                    .subbinary_from_original(
                        subbinary.original(),
                        subbinary.byte_offset() + index,
                        subbinary.bit_offset(),
                        0,
                        0,
                    )?
                    .encode()?;

                // Don't make a subbinary of the suffix since it is the same as the
                // `binary`.
                heap.tuple_from_slice(&[empty_prefix, binary])
                    .map_err(|e| e.into())
                    .and_then(|t| t.encode())
            } else {
                // total_byte_length includes +1 byte if bits
                let total_byte_length = subbinary.total_byte_len();

                if index < total_byte_length {
                    let mut heap = process.acquire_heap();
                    let original = subbinary.original();
                    let byte_offset = subbinary.byte_offset();
                    let bit_offset = subbinary.bit_offset();

                    let prefix = heap
                        .subbinary_from_original(original, byte_offset, bit_offset, index, 0)?
                        .encode()?;
                    let suffix = heap
                        .subbinary_from_original(
                            original,
                            byte_offset + index,
                            bit_offset,
                            // full_byte_count does not include bits
                            subbinary.full_byte_len() - index,
                            subbinary.partial_byte_bit_len(),
                        )?
                        .encode()?;

                    heap.tuple_from_slice(&[prefix, suffix])
                        .map_err(|e| e.into())
                        .and_then(|t| t.encode())
                } else if (index == total_byte_length) & (subbinary.partial_byte_bit_len() == 0) {
                    let mut heap = process.acquire_heap();
                    let empty_suffix = heap
                        .subbinary_from_original(
                            subbinary.original(),
                            subbinary.byte_offset() + index,
                            subbinary.bit_offset(),
                            0,
                            0,
                        )?
                        .encode()?;

                    heap.tuple_from_slice(&[binary, empty_suffix])
                        .map_err(|e| e.into())
                        .and_then(|t| t.encode())
                } else {
                    Err(badarg!().into())
                }
            }
        }
        _ => Err(badarg!().into()),
    }
}
