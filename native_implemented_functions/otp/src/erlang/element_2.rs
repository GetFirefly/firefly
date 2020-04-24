// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use anyhow::*;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::term::prelude::*;

use native_implemented_function::native_implemented_function;

use crate::runtime::context::*;

/// `element/2`
#[native_implemented_function(element/2)]
pub fn result(index: Term, tuple: Term) -> exception::Result<Term> {
    let tuple_tuple = term_try_into_tuple!(tuple)?;
    let one_based_index = term_try_into_one_based_index(index)?;

    tuple_tuple
        .get_element(one_based_index)
        .with_context(|| format!("index ({})", index))
        .map_err(From::from)
}
