// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Term;

use native_implemented_function::native_implemented_function;

/// `-/2` infix operator
#[native_implemented_function(-/2)]
pub fn result(process: &Process, minuend: Term, subtrahend: Term) -> exception::Result<Term> {
    number_infix_operator!(minuend, subtrahend, process, checked_sub, -)
}
