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

use crate::erlang::iolist_or_binary::{self, *};

/// Returns the size, in bytes, of the binary that would be result from iolist_to_binary/1
#[native_implemented_function(iolist_size/1)]
pub fn result(process: &Process, iolist_or_binary: Term) -> exception::Result<Term> {
    iolist_or_binary::result(process, iolist_or_binary, iolist_or_binary_size)
}

fn iolist_or_binary_size(process: &Process, iolist_or_binary: Term) -> exception::Result<Term> {
    let mut size: usize = 0;
    let mut stack: Vec<Term> = vec![iolist_or_binary];

    while let Some(top) = stack.pop() {
        match top.decode()? {
            TypedTerm::SmallInteger(small_integer) => {
                let _: u8 = small_integer
                    .try_into()
                    .with_context(|| element_type_context(iolist_or_binary, top))?;

                size += 1;
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
                                "iolist_or_binary ({}) tail ({}) cannot be a byte",
                                iolist_or_binary, tail
                            ))
                            .map_err(From::from)
                    }
                    Err(_) => stack.push(tail),
                };

                stack.push(boxed_cons.head);
            }
            TypedTerm::HeapBinary(heap_binary) => size += heap_binary.full_byte_len(),
            TypedTerm::MatchContext(match_context) => {
                if match_context.is_binary() {
                    size += match_context.full_byte_len();
                } else {
                    return Err(NotABinary)
                        .context(element_not_a_binary_context(iolist_or_binary, top))
                        .map_err(From::from);
                }
            }
            TypedTerm::ProcBin(proc_bin) => size += proc_bin.total_byte_len(),
            TypedTerm::SubBinary(subbinary) => {
                if subbinary.is_binary() {
                    size += subbinary.full_byte_len();
                } else {
                    return Err(NotABinary)
                        .context(element_not_a_binary_context(iolist_or_binary, top))
                        .map_err(From::from);
                }
            }
            _ => {
                return Err(TypeError)
                    .context(element_type_context(iolist_or_binary, top))
                    .map_err(From::from);
            }
        }
    }

    process.integer(size).map_err(From::from)
}
