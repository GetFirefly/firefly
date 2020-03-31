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
use liblumen_alloc::erts::term::index::OneBasedIndex;
use liblumen_alloc::erts::term::prelude::*;

use native_implemented_function::native_implemented_function;

use crate::runtime::context::*;

#[native_implemented_function(setelement/3)]
pub fn native(process: &Process, index: Term, tuple: Term, value: Term) -> exception::Result<Term> {
    let initial_inner_tuple = term_try_into_tuple!(tuple)?;
    let length = initial_inner_tuple.len();
    let index_zero_based: OneBasedIndex = index
        .try_into()
        .with_context(|| term_is_not_in_one_based_range(index, length))?;

    if index_zero_based < length {
        if index_zero_based == 0 {
            if 1 < length {
                process.tuple_from_slices(&[&[value], &initial_inner_tuple[1..]])
            } else {
                process.tuple_from_slice(&[value])
            }
        } else if index_zero_based < (length - 1) {
            process.tuple_from_slices(&[
                &initial_inner_tuple[..index_zero_based],
                &[value],
                &initial_inner_tuple[(index_zero_based + 1)..],
            ])
        } else {
            process.tuple_from_slices(&[&initial_inner_tuple[..index_zero_based], &[value]])
        }
        .map_err(|error| error.into())
    } else {
        Err(TryIntoIntegerError::OutOfRange)
            .with_context(|| term_is_not_in_one_based_range(index, length))
            .map_err(From::from)
    }
}
