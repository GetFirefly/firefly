// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::exception::Exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::binary::Bitstring;
use liblumen_alloc::erts::term::{Term, TypedTerm};
use lumen_runtime_macros::native_implemented_function;

/// Returns the size, in bytes, of the binary that would be result from iolist_to_binary/1
#[native_implemented_function(iolist_size/1)]
pub fn native(process: &Process, iolist_or_binary: Term) -> exception::Result {
    let mut stack: Vec<Term> = vec![iolist_or_binary];
    match size(process, &mut stack, 0) {
        Ok(size) => Ok(process.integer(size).unwrap()),
        Err(bad) => Err(bad),
    }
}

fn size(process: &Process, vec: &mut Vec<Term>, acc: usize) -> Result<usize, Exception> {
    if let Some(term) = vec.pop() {
        match term.to_typed_term().unwrap() {
            TypedTerm::SmallInteger(_) => size(process, vec, acc + 1),

            TypedTerm::Nil => size(process, vec, acc),

            TypedTerm::Boxed(boxed) => match boxed.to_typed_term().unwrap() {
                TypedTerm::HeapBinary(heap_binary) => {
                    size(process, vec, acc + heap_binary.full_byte_len())
                }

                TypedTerm::ProcBin(proc_binary) => {
                    size(process, vec, acc + proc_binary.full_byte_len())
                }

                TypedTerm::SubBinary(subbinary) => {
                    size(process, vec, acc + subbinary.full_byte_len())
                }

                _ => Err(badarg!().into()),
            },

            TypedTerm::List(boxed_cons) => {
                if boxed_cons.tail.is_smallint() {
                    Err(badarg!().into())
                } else {
                    vec.push(boxed_cons.tail);
                    vec.push(boxed_cons.head);
                    size(process, vec, acc)
                }
            }

            _ => Err(badarg!().into()),
        }
    } else {
        Ok(acc)
    }
}
