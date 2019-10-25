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

use crate::otp::erlang;

#[native_implemented_function(binary_part/2)]
pub fn native(process: &Process, binary: Term, start_length: Term) -> exception::Result {
    let option_result = match start_length.to_typed_term().unwrap() {
        TypedTerm::Boxed(unboxed_start_length) => {
            match unboxed_start_length.to_typed_term().unwrap() {
                TypedTerm::Tuple(tuple) => {
                    if tuple.len() == 2 {
                        Some(erlang::binary_part_3::native(
                            process, binary, tuple[0], tuple[1],
                        ))
                    } else {
                        None
                    }
                }
                _ => None,
            }
        }
        _ => None,
    };

    match option_result {
        Some(result) => result,
        None => Err(badarg!().into()),
    }
}
