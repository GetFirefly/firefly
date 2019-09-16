// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::term::{Boxed, Term, Tuple};

use lumen_runtime_macros::native_implemented_function;

/// `element/2`
#[native_implemented_function(element/2)]
pub fn native(index: Term, tuple: Term) -> exception::Result {
    let tuple_tuple: Boxed<Tuple> = tuple.try_into()?;

    tuple_tuple
        .get_element_from_one_based_term_index(index)
        .map_err(|error| error.into())
}
