// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::Term;

use lumen_runtime_macros::native_implemented_function;

use crate::otp::erlang::float_to_string::float_to_string;

#[native_implemented_function(float_to_binary/1)]
pub fn native(process: &Process, float: Term) -> exception::Result {
    float_to_string(float, Default::default())
        .map_err(|error| error.into())
        .and_then(|string| {
            process
                .binary_from_str(&string)
                .map_err(|alloc| alloc.into())
        })
}
