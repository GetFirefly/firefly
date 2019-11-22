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

use lumen_runtime_macros::native_implemented_function;

/// `element/2`
#[native_implemented_function(element/2)]
pub fn native(index: Term, tuple: Term) -> exception::Result<Term> {
    let tuple_tuple: Boxed<Tuple> = tuple.try_into().context("tuple must be a tuple")?;
    let index: OneBasedIndex = index
        .try_into()
        .context("index must be a non-negative integer")?;

    tuple_tuple
        .get_element(index)
        .context("index out of bounds")
        .map_err(From::from)
}
