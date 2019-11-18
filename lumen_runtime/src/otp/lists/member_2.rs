// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::term::prelude::*;

use lumen_runtime_macros::native_implemented_function;

#[native_implemented_function(member/2)]
pub fn native(element: Term, list: Term) -> exception::Result<Term> {
    match list.decode()? {
        TypedTerm::Nil => Ok(false.into()),
        TypedTerm::List(cons) => {
            for result in cons.into_iter() {
                match result {
                    Ok(term) => {
                        if term == element {
                            return Ok(true.into());
                        }
                    }
                    Err(_) => return Err(badarg!().into()),
                };
            }

            Ok(false.into())
        }
        _ => Err(badarg!().into()),
    }
}
