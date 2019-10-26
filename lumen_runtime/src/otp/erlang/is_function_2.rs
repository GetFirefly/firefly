// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::term::prelude::Term;

use lumen_runtime_macros::native_implemented_function;

#[native_implemented_function(is_function/2)]
fn native(term: Term, arity: Term) -> exception::Result {
    let arity_arity: usize = arity.try_into()?;

    Ok(term.is_function_with_arity(arity_arity).into())
}
