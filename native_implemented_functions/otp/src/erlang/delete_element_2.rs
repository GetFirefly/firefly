// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;

use anyhow::*;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use native_implemented_function::native_implemented_function;

use crate::runtime::context::*;

/// `delete_element/2`
#[native_implemented_function(delete_element/2)]
pub fn result(process: &Process, index: Term, tuple: Term) -> exception::Result<Term> {
    let initial_inner_tuple = term_try_into_tuple!(tuple)?;
    let initial_len = initial_inner_tuple.len();

    if initial_len > 0 {
        let index_zero_based: OneBasedIndex = index
            .try_into()
            .with_context(|| term_is_not_in_one_based_range(index, initial_len))?;

        if index_zero_based < initial_len {
            let smaller_len = initial_len - 1;
            let smaller_element_iterator =
                initial_inner_tuple
                    .iter()
                    .enumerate()
                    .filter_map(|(old_index, old_term)| {
                        if index_zero_based == old_index {
                            None
                        } else {
                            Some(*old_term)
                        }
                    });
            let smaller_tuple = process.tuple_from_iter(smaller_element_iterator, smaller_len)?;

            Ok(smaller_tuple)
        } else {
            Err(TryIntoIntegerError::OutOfRange)
                .with_context(|| term_is_not_in_one_based_range(index, initial_len))
                .map_err(From::from)
        }
    } else {
        Err(anyhow!("tuple ({}) is empty", tuple).into())
    }
}
