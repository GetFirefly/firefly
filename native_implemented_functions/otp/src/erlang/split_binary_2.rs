// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;

use anyhow::*;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::alloc::TermAlloc;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use native_implemented_function::native_implemented_function;

#[native_implemented_function(split_binary/2)]
pub fn result(process: &Process, binary: Term, position: Term) -> exception::Result<Term> {
    let index: usize = position
        .try_into()
        .with_context(|| format!("position ({}) must be in 0..byte_size(binary)", position))?;

    match binary.decode().unwrap() {
        binary_box @ TypedTerm::HeapBinary(_) | binary_box @ TypedTerm::ProcBin(_) => {
            if index == 0 {
                let mut heap = process.acquire_heap();

                let empty_prefix = heap
                    .subbinary_from_original(binary, index, 0, 0, 0)?
                    .encode()?;

                // Don't make a subbinary of the suffix since it is the same as the
                // `binary`.
                let boxed_tuple = heap.tuple_from_slice(&[empty_prefix, binary])?;
                let tuple_term = boxed_tuple.encode()?;

                Ok(tuple_term)
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

                    let boxed_tuple = heap.tuple_from_slice(&[prefix, suffix])?;
                    let tuple_term = boxed_tuple.encode()?;

                    Ok(tuple_term)
                } else if index == full_byte_length {
                    let mut heap = process.acquire_heap();
                    let empty_suffix = heap
                        .subbinary_from_original(binary, index, 0, 0, 0)?
                        .encode()?;

                    // Don't make a subbinary of the prefix since it is the same as the
                    // `binary`.
                    let boxed_tuple = heap.tuple_from_slice(&[binary, empty_suffix])?;
                    let tuple_term = boxed_tuple.encode()?;

                    Ok(tuple_term)
                } else {
                    Err(anyhow!(
                        "index ({}) exceeds full byte length ({})",
                        index,
                        full_byte_length
                    )
                    .into())
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
                let boxed_tuple = heap.tuple_from_slice(&[empty_prefix, binary])?;
                let tuple_term = boxed_tuple.encode()?;

                Ok(tuple_term)
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

                    let boxed_tuple = heap.tuple_from_slice(&[prefix, suffix])?;
                    let tuple_term = boxed_tuple.encode()?;

                    Ok(tuple_term)
                } else if index == total_byte_length {
                    let partial_byte_bit_len = subbinary.partial_byte_bit_len();

                    if partial_byte_bit_len == 0 {
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

                        let boxed_tuple = heap.tuple_from_slice(&[binary, empty_suffix])?;
                        let tuple_term = boxed_tuple.encode()?;

                        Ok(tuple_term)
                    } else {
                        Err(anyhow!("bitstring ({}) has {} bits in its partial bytes, so the index ({}) cannot equal the total byte length ({})", binary, partial_byte_bit_len, index, total_byte_length).into())
                    }
                } else {
                    Err(anyhow!(
                        "index ({}) exceeds total byte length ({}) of bitstring ({})",
                        index,
                        total_byte_length,
                        binary
                    )
                    .into())
                }
            }
        }
        _ => Err(anyhow!(TypeError))
            .context(format!("binary ({}) is not a bitstring", binary))
            .map_err(From::from),
    }
}
