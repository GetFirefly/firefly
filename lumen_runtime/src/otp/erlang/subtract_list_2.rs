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

/// `--/2`
#[native_implemented_function(--/2)]
pub fn native(process: &Process, minuend: Term, subtrahend: Term) -> exception::Result<Term> {
    match (minuend.decode().unwrap(), subtrahend.decode().unwrap()) {
        (TypedTerm::Nil, TypedTerm::Nil) => Ok(minuend),
        (TypedTerm::Nil, TypedTerm::List(subtrahend_cons)) => {
            if subtrahend_cons.is_proper() {
                Ok(minuend)
            } else {
                Err(badarg!().into())
            }
        }
        (TypedTerm::List(minuend_cons), TypedTerm::Nil) => {
            if minuend_cons.is_proper() {
                Ok(minuend)
            } else {
                Err(badarg!().into())
            }
        }
        (TypedTerm::List(minuend_cons), TypedTerm::List(subtrahend_cons)) => {
            match minuend_cons
                .into_iter()
                .collect::<std::result::Result<Vec<Term>, _>>()
            {
                Ok(mut minuend_vec) => {
                    for result in subtrahend_cons.into_iter() {
                        match result {
                            Ok(subtrahend_element) => minuend_vec.remove_item(&subtrahend_element),
                            Err(ImproperList { .. }) => return Err(badarg!().into()),
                        };
                    }

                    process
                        .list_from_slice(&minuend_vec)
                        .map_err(|error| error.into())
                }
                Err(ImproperList { .. }) => Err(badarg!().into()),
            }
        }
        _ => Err(badarg!().into()),
    }
}
