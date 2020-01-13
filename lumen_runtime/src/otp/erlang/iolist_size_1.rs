// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use anyhow::*;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::exception::Exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use lumen_runtime_macros::native_implemented_function;

/// Returns the size, in bytes, of the binary that would be result from iolist_to_binary/1
#[native_implemented_function(iolist_size/1)]
pub fn native(process: &Process, iolist_or_binary: Term) -> exception::Result<Term> {
    let mut stack: Vec<Term> = vec![iolist_or_binary];
    match size(process, iolist_or_binary, &mut stack, 0) {
        Ok(size) => Ok(process.integer(size).unwrap()),
        Err(bad) => Err(bad),
    }
}

fn size(
    process: &Process,
    iolist_or_binary: Term,
    vec: &mut Vec<Term>,
    acc: usize,
) -> Result<usize, Exception> {
    if let Some(term) = vec.pop() {
        match term.decode().unwrap() {
            TypedTerm::SmallInteger(_) => size(process, iolist_or_binary, vec, acc + 1),

            TypedTerm::Nil => size(process, iolist_or_binary, vec, acc),

            TypedTerm::HeapBinary(heap_binary) => size(
                process,
                iolist_or_binary,
                vec,
                acc + heap_binary.full_byte_len(),
            ),

            TypedTerm::ProcBin(proc_binary) => size(
                process,
                iolist_or_binary,
                vec,
                acc + proc_binary.full_byte_len(),
            ),

            TypedTerm::SubBinary(subbinary) => size(
                process,
                iolist_or_binary,
                vec,
                acc + subbinary.full_byte_len(),
            ),

            TypedTerm::List(boxed_cons) => {
                let tail = boxed_cons.tail;

                if tail.is_smallint() {
                    Err(TypeError)
                        .context(format!(
                            "iolist_or_binary ({}) tail ({}) cannot be a byte",
                            iolist_or_binary, tail
                        ))
                        .map_err(From::from)
                } else {
                    vec.push(boxed_cons.tail);
                    vec.push(boxed_cons.head);
                    size(process, iolist_or_binary, vec, acc)
                }
            }

            _ => Err(TypeError)
                .context(format!(
                    "iolist_or_binary ({}) is not an iolist or a binary",
                    iolist_or_binary
                ))
                .map_err(From::from),
        }
    } else {
        Ok(acc)
    }
}
