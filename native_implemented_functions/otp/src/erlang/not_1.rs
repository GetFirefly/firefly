// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::term::prelude::Term;

use native_implemented_function::native_implemented_function;

use crate::runtime::context::*;

/// `not/1` prefix operator.
#[native_implemented_function(not/1)]
pub fn native(boolean: Term) -> exception::Result<Term> {
    let boolean_bool: bool = term_try_into_bool("boolean", boolean)?;
    let output = !boolean_bool;

    Ok(output.into())
}
