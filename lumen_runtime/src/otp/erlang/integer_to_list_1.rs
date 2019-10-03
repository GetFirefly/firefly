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

use crate::otp::erlang::integer_to_string::decimal_integer_to_string;

#[native_implemented_function(integer_to_list/1)]
pub fn native(process: &Process, integer: Term) -> exception::Result {
    decimal_integer_to_string(integer).and_then(|string| {
        process
            .list_from_chars(string.chars())
            .map_err(|alloc| alloc.into())
    })
}
