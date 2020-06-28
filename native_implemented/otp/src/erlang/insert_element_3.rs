#[cfg(all(not(any(target_arch = "wasm32", feature = "runtime_minimal")), test))]
mod test;

use std::convert::TryInto;

use anyhow::*;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::index::OneBasedIndex;
use liblumen_alloc::erts::term::prelude::*;

use crate::runtime::context::*;

#[native_implemented::function(insert_element/3)]
pub fn result(
    process: &Process,
    index: Term,
    tuple: Term,
    element: Term,
) -> exception::Result<Term> {
    let initial_inner_tuple = term_try_into_tuple!(tuple)?;
    let length = initial_inner_tuple.len();
    let index_one_based: OneBasedIndex = index
        .try_into()
        .with_context(|| term_is_not_in_one_based_range(index, length + 1))?;

    // can be equal to arity when insertion is at the end
    if index_one_based <= length {
        if index_one_based == 0 {
            process.tuple_from_slices(&[&[element], &initial_inner_tuple[..]])
        } else if index_one_based < length {
            process.tuple_from_slices(&[
                &initial_inner_tuple[..index_one_based],
                &[element],
                &initial_inner_tuple[index_one_based..],
            ])
        } else {
            process.tuple_from_slices(&[&initial_inner_tuple[..], &[element]])
        }
        .map_err(From::from)
    } else {
        Err(TryIntoIntegerError::OutOfRange)
            .with_context(|| term_is_not_in_one_based_range(index, length + 1))
            .map_err(From::from)
    }
}
