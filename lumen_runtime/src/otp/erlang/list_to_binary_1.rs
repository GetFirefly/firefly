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

use lumen_runtime_macros::native_implemented_function;

#[native_implemented_function(list_to_binary/1)]
pub fn native(process: &Process, iolist: Term) -> exception::Result<Term> {
    match iolist.decode()? {
        TypedTerm::Nil | TypedTerm::List(_) => {
            let mut byte_vec: Vec<u8> = Vec::new();
            let mut stack: Vec<Term> = vec![iolist];

            while let Some(top) = stack.pop() {
                match top.decode()? {
                    TypedTerm::SmallInteger(small_integer) => {
                        let top_byte = small_integer
                            .try_into()
                            .with_context(|| element_context(iolist, top))?;

                        byte_vec.push(top_byte);
                    }
                    TypedTerm::Nil => (),
                    TypedTerm::List(boxed_cons) => {
                        // @type iolist :: maybe_improper_list(byte() | binary() | iolist(),
                        // binary() | []) means that `byte()` isn't allowed
                        // for `tail`s unlike `head`.

                        let tail = boxed_cons.tail;
                        let result_u8: Result<u8, _> = tail.try_into();

                        match result_u8 {
                            Ok(_) => {
                                return Err(TypeError)
                                    .context(format!(
                                        "iolist ({}) tail ({}) cannot be a byte",
                                        iolist, tail
                                    ))
                                    .map_err(From::from)
                            }
                            Err(_) => stack.push(tail),
                        };

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
                            return Err(NotABinary)
                                .context(element_context(iolist, top))
                                .map_err(From::from);
                        }
                    }
                    _ => {
                        return Err(TypeError)
                            .context(element_context(iolist, top))
                            .map_err(From::from)
                    }
                }
            }

            Ok(process.binary_from_bytes(byte_vec.as_slice()).unwrap())
        }
        _ => Err(TypeError)
            .context(format!("iolist ({}) is not a list", iolist))
            .map_err(From::from),
    }
}

fn element_context(iolist: Term, element: Term) -> String {
    format!(
        "iolist ({}) element ({}) is not a byte, binary, or nested iolist",
        iolist, element
    )
}
