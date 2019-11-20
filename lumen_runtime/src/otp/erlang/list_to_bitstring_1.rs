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
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use lumen_runtime_macros::native_implemented_function;

#[native_implemented_function(list_to_bitstring/1)]
pub fn native(process: &Process, iolist: Term) -> exception::Result<Term> {
    match iolist.decode().unwrap() {
        TypedTerm::Nil | TypedTerm::List(_) => {
            let mut byte_vec: Vec<u8> = Vec::new();
            let mut partial_byte_bit_count = 0;
            let mut partial_byte = 0;
            let mut stack: Vec<Term> = vec![iolist];

            while let Some(top) = stack.pop() {
                match top.decode().unwrap() {
                    TypedTerm::SmallInteger(small_integer) => {
                        let top_byte = small_integer
                            .try_into()
                            .context("list element does not fit in a byte")?;

                        if partial_byte_bit_count == 0 {
                            byte_vec.push(top_byte);
                        } else {
                            partial_byte |= top_byte >> partial_byte_bit_count;
                            byte_vec.push(partial_byte);

                            partial_byte = top_byte << (8 - partial_byte_bit_count);
                        }
                    }
                    TypedTerm::Nil => (),
                    TypedTerm::List(boxed_cons) => {
                        // @type bitstring_list ::
                        //   maybe_improper_list(byte() | bitstring() | bitstring_list(),
                        //                       bitstring() | [])
                        // means that `byte()` isn't allowed for `tail`s unlike `head`.

                        let tail = boxed_cons.tail;

                        if tail.is_smallint() {
                            return Err(badarg!(process).into());
                        } else {
                            stack.push(tail);
                        }

                        stack.push(boxed_cons.head);
                    }
                    TypedTerm::HeapBinary(heap_binary) => {
                        if partial_byte_bit_count == 0 {
                            byte_vec.extend_from_slice(heap_binary.as_bytes());
                        } else {
                            for byte in heap_binary.as_bytes() {
                                partial_byte |= byte >> partial_byte_bit_count;
                                byte_vec.push(partial_byte);

                                partial_byte = byte << (8 - partial_byte_bit_count);
                            }
                        }
                    }
                    TypedTerm::SubBinary(subbinary) => {
                        if partial_byte_bit_count == 0 {
                            if subbinary.is_aligned() {
                                byte_vec.extend(unsafe { subbinary.as_bytes_unchecked() });
                            } else {
                                byte_vec.extend(subbinary.full_byte_iter());
                            }
                        } else {
                            for byte in subbinary.full_byte_iter() {
                                partial_byte |= byte >> partial_byte_bit_count;
                                byte_vec.push(partial_byte);

                                partial_byte = byte << (8 - partial_byte_bit_count);
                            }
                        }

                        if !subbinary.is_binary() {
                            for bit in subbinary.partial_byte_bit_iter() {
                                partial_byte |= bit << (7 - partial_byte_bit_count);

                                if partial_byte_bit_count == 7 {
                                    byte_vec.push(partial_byte);
                                    partial_byte_bit_count = 0;
                                    partial_byte = 0;
                                } else {
                                    partial_byte_bit_count += 1;
                                }
                            }
                        }
                    }
                    _ => return Err(badarg!(process).into()),
                }
            }

            if partial_byte_bit_count == 0 {
                Ok(process.binary_from_bytes(byte_vec.as_slice()).unwrap())
            } else {
                let full_byte_len = byte_vec.len();
                byte_vec.push(partial_byte);
                let original = process.binary_from_bytes(byte_vec.as_slice()).unwrap();

                Ok(process
                    .subbinary_from_original(original, 0, 0, full_byte_len, partial_byte_bit_count)
                    .unwrap())
            }
        }
        _ => Err(badarg!(process).into()),
    }
}
