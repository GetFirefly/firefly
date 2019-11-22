// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;

use anyhow::*;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::Arity;

use lumen_runtime_macros::native_implemented_function;

#[native_implemented_function(is_function/2)]
fn native(term: Term, arity: Term) -> exception::Result<Term> {
    let arity_arity: Arity = arity
        .try_into()
        .context("arity must be an integer in 0-255")?;

    Ok(term.decode()?.is_function_with_arity(arity_arity).into())
}
