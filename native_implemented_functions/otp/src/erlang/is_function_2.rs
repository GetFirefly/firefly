// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::term::prelude::*;

use native_implemented_function::native_implemented_function;

use crate::runtime::context::*;

#[native_implemented_function(is_function/2)]
fn native(term: Term, arity: Term) -> exception::Result<Term> {
    let arity_arity = term_try_into_arity(arity)?;

    Ok(term.decode()?.is_function_with_arity(arity_arity).into())
}
