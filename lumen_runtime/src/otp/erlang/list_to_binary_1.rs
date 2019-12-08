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

#[native_implemented_function(list_to_binary/1)]
pub fn native(process: &Process, iolist: Term) -> exception::Result<Term> {
    match iolist.decode().unwrap() {
        TypedTerm::Nil | TypedTerm::List(_) => {
            let mut byte_vec: Vec<u8> = Vec::new();
            let mut stack: Vec<Term> = vec![iolist];

            while let Some(top) = stack.pop() {
                match top.decode().unwrap() {
                    TypedTerm::SmallInteger(small_integer) => {
                        let top_byte = small_integer.try_into()?;

                        byte_vec.push(top_byte);
                    }
                    TypedTerm::Nil => (),
                    TypedTerm::List(boxed_cons) => {
                        // @type iolist :: maybe_improper_list(byte() | binary() | iolist(),
                        // binary() | []) means that `byte()` isn't allowed
                        // for `tail`s unlike `head`.

                        let tail = boxed_cons.tail;

                        if tail.is_smallint() {
                            return Err(badarg!().into());
                        } else {
                            stack.push(tail);
                        }

                        stack.push(boxed_cons.head);
                    }
                    TypedTerm::HeapBinary(heap_binary) => {
                        byte_vec.extend_from_slice(heap_binary.as_bytes());
                    }
                    TypedTerm::SubBinary(subbinary) => {
                        if subbinary.is_binary() {
                            if subbinary.is_aligned() {
                                byte_vec.extend(unsafe { subbinary.as_bytes_unchecked() });
                            } else {
                                byte_vec.extend(subbinary.full_byte_iter());
                            }
                        } else {
                            return Err(badarg!().into());
                        }
                    }
                    TypedTerm::ProcBin(procbin) => {
                        byte_vec.extend_from_slice(procbin.as_bytes());
                    }
                    _ => return Err(badarg!().into())
                }
            }

            Ok(process.binary_from_bytes(byte_vec.as_slice()).unwrap())
        }
        _ => Err(badarg!().into()),
    }
}
