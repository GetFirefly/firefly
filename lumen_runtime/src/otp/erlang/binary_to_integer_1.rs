// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;

use num_bigint::BigInt;

use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::Term;

use lumen_runtime_macros::native_implemented_function;

#[native_implemented_function(binary_to_integer/1)]
pub fn native(process: &Process, binary: Term) -> exception::Result {
    let string: String = binary.try_into()?;

    match BigInt::parse_bytes(string.as_bytes(), 10) {
        Some(big_int) => process.integer(big_int).map_err(|error| error.into()),
        None => Err(badarg!().into()),
    }
}
