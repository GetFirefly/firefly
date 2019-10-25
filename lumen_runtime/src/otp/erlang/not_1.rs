// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::term::prelude::Term;(

use lumen_runtime_macros::native_implemented_function;

/// `not/1` prefix operator.
#[native_implemented_function(not/1)]
pub fn native(boolean: Term) -> exception::Result {
    let boolean_bool: bool = boolean.try_into()?;
    let output = !boolean_bool;

    Ok(output.into())
}
