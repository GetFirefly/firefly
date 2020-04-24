// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Term;

use native_implemented_function::native_implemented_function;

use crate::erlang::unique_integer::{unique_integer, Options};

#[native_implemented_function(unique_integer/1)]
pub fn result(process: &Process, options: Term) -> exception::Result<Term> {
    let options_options: Options = options.try_into()?;

    unique_integer(process, options_options)
}
