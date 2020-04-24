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

use crate::erlang::term_to_binary::term_to_binary;

#[native_implemented_function(term_to_binary/1)]
pub fn result(process: &Process, term: Term) -> exception::Result<Term> {
    term_to_binary(process, term, Default::default())
}
