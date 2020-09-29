use std::convert::TryInto;

use anyhow::*;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::runtime::context::*;

/// `delete_element/2`
#[native_implemented::function(erlang:delete_element/2)]
pub fn result(process: &Process, index: Term, tuple: Term) -> exception::Result<Term> {
    let initial_inner_tuple = term_try_into_tuple!(tuple)?;
    let initial_len = initial_inner_tuple.len();

    if initial_len > 0 {
        let index_one_based: OneBasedIndex = index
            .try_into()
            .with_context(|| term_is_not_in_one_based_range(index, initial_len))?;
        let index_zero_based: usize = index_one_based.into();

        if index_zero_based < initial_len {
            let mut new_elements_vec = initial_inner_tuple[..].to_vec();
            new_elements_vec.remove(index_zero_based);
            let smaller_tuple = process.tuple_from_slice(&new_elements_vec);

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
