// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Term;

use lumen_runtime_macros::native_implemented_function;

use crate::binary_to_string::binary_to_string;
use crate::otp::erlang::string_to_float::string_to_float;

#[native_implemented_function(binary_to_float/1)]
pub fn native(process: &Process, binary: Term) -> exception::Result<Term> {
    binary_to_string(binary)
        .and_then(|string| string_to_float(process, string.as_str()))
        .map_err(|_| badarg!(process).into())
}
