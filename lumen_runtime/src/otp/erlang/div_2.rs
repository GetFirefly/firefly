// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Term;

use lumen_runtime_macros::native_implemented_function;

/// `div/2` infix operator.  Integer division.
#[native_implemented_function(div/2)]
pub fn native(process: &Process, dividend: Term, divisor: Term) -> exception::Result {
    integer_infix_operator!(dividend, divisor, process, /)
}
