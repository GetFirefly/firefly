// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::Term;

use lumen_runtime_macros::native_implemented_function;

use crate::otp::erlang::number_to_integer::{f64_to_integer, NumberToInteger};

#[native_implemented_function(floor/1)]
pub fn native(process: &Process, number: Term) -> exception::Result {
    match number.into() {
        NumberToInteger::Integer(integer) => Ok(integer),
        NumberToInteger::F64(f) => {
            let floor = f.floor();

            f64_to_integer(process, floor)
        }
        NumberToInteger::NotANumber => Err(badarg!().into()),
    }
}
