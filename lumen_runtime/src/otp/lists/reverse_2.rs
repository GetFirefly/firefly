// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use lumen_runtime_macros::native_implemented_function;

#[native_implemented_function(reverse/2)]
pub fn native(process: &Process, list: Term, tail: Term) -> exception::Result {
    match list.decode().unwrap() {
        TypedTerm::Nil => Ok(tail),
        TypedTerm::List(cons) => {
            let mut reversed = tail;

            for result in cons.into_iter() {
                match result {
                    Ok(element) => {
                        reversed = process.cons(element, reversed)?;
                    }
                    Err(_) => return Err(badarg!().into()),
                }
            }

            Ok(reversed)
        }
        _ => Err(badarg!().into()),
    }
}
