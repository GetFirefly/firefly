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

use crate::otp::erlang::binary_to_term_2;

#[native_implemented_function(binary_to_term/1)]
pub fn native(process: &Process, binary: Term) -> exception::Result {
    binary_to_term_2(binary, Term::NIL, process)
}
