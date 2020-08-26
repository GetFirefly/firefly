#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;

use anyhow::*;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::index::OneBasedIndex;
use liblumen_alloc::erts::term::prelude::*;

use crate::runtime::context::*;

#[native_implemented::function(erlang:setelement/3)]
pub fn result(process: &Process, index: Term, tuple: Term, value: Term) -> exception::Result<Term> {
    let initial_inner_tuple = term_try_into_tuple!(tuple)?;
    let length = initial_inner_tuple.len();
    let index_one_based: OneBasedIndex = index
        .try_into()
        .with_context(|| term_is_not_in_one_based_range(index, length))?;
    let index_zero_based: usize = index_one_based.into();

    if index_zero_based < length {
        let mut final_element_vec = initial_inner_tuple[..].to_vec();
        final_element_vec[index_zero_based] = value;
        let final_tuple = process.tuple_from_slice(&final_element_vec);

        Ok(final_tuple)
    } else {
        Err(TryIntoIntegerError::OutOfRange)
            .with_context(|| term_is_not_in_one_based_range(index, length))
            .map_err(From::from)
    }
}
