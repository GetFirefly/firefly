// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::{Term, TypedTerm};

use lumen_runtime_macros::native_implemented_function;

#[native_implemented_function(integer_to_binary/1)]
pub fn native(process: &Process, integer: Term) -> exception::Result {
    let option_string: Option<String> = match integer.to_typed_term().unwrap() {
        TypedTerm::SmallInteger(small_integer) => Some(small_integer.to_string()),
        TypedTerm::Boxed(boxed) => match boxed.to_typed_term().unwrap() {
            TypedTerm::BigInteger(big_integer) => Some(big_integer.to_string()),
            _ => None,
        },
        _ => None,
    };

    match option_string {
        Some(string) => process
            .binary_from_str(&string)
            .map_err(|alloc| alloc.into()),
        None => Err(badarg!().into()),
    }
}
