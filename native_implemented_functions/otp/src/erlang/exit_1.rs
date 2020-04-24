// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use anyhow::*;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::term::prelude::Term;
use liblumen_alloc::exit;

use native_implemented_function::native_implemented_function;

#[native_implemented_function(exit/1)]
fn result(reason: Term) -> exception::Result<Term> {
    Err(exit!(reason, anyhow!("explicit exit from Erlang").into()).into())
}
