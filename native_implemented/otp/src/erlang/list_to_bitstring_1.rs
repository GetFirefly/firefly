#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;
use std::ptr::NonNull;

use anyhow::*;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::Term;

#[native_implemented::function(erlang:list_to_bitstring/1)]
pub fn result(process: &Process, bitstring_list: Term) -> Result<Term, NonNull<ErlangException>> {
    match bitstring_list {
        Term::Nil | Term::Cons(_) => {
            let mut byte_vec: Vec<u8> = Vec::new();
            let mut partial_byte_bit_count = 0;
            let mut partial_byte = 0;
            let mut stack: Vec<Term> = vec![bitstring_list];

            while let Some(top) = stack.pop() {
                match top {
                    Term::Int(small_integer) => {
                        let top_byte = small_integer
                            .try_into()
                            .context(element_context(bitstring_list, top))?;

                        if partial_byte_bit_count == 0 {
                            byte_vec.push(top_byte);
                        } else {
                            partial_byte |= top_byte >> partial_byte_bit_count;
                            byte_vec.push(partial_byte);

                            partial_byte = top_byte << (8 - partial_byte_bit_count);
                        }
                    }
                    Term::Nil => (),
                    Term::Cons(non_null_cons) => {
                        let cons = unsafe { non_null_cons.as_ref() };
                        // @type bitstring_list ::
                        //   maybe_improper_list(byte() | bitstring() | bitstring_list(),
                        //                       bitstring() | [])
                        // means that `byte()` isn't allowed for `tail`s unlike `head`.

                        let tail = cons.tail();
                        let result_u8: Result<u8, _> = tail.try_into();

                        match result_u8 {
                            Ok(_) => {
                                return Err(TypeError)
                                    .context(format!(
                                        "bitstring_list ({}) tail ({}) cannot be a byte",
                                        bitstring_list, tail
                                    ))
                                    .map_err(From::from)
                            }
                            Err(_) => stack.push(tail),
                        };

                        stack.push(cons.head());
                    }
                    Term::HeapBinary(heap_binary) => {
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
                    Term::RefBinary(subbinary) => {
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
                    _ => {
                        return Err(TypeError)
                            .context(element_context(bitstring_list, top))
                            .map_err(From::from)
                    }
                }
            }

            if partial_byte_bit_count == 0 {
                Ok(process.binary_from_bytes(byte_vec.as_slice()))
            } else {
                let full_byte_len = byte_vec.len();
                byte_vec.push(partial_byte);
                let original = process.binary_from_bytes(byte_vec.as_slice());

                Ok(process.subbinary_from_original(
                    original,
                    0,
                    0,
                    full_byte_len,
                    partial_byte_bit_count,
                ))
            }
        }
        _ => Err(TypeError)
            .context(format!("bitstring_list ({}) is not a list", bitstring_list))
            .map_err(From::from),
    }
}

fn element_context(bitstring_list: Term, element: Term) -> String {
    format!(
        "bitstring_list ({}) element ({}) is not a byte, bitstring, or nested bitstring_list",
        bitstring_list, element
    )
}
