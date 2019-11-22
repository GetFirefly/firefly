// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;

use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::index::OneBasedIndex;
use liblumen_alloc::erts::term::prelude::*;

use lumen_runtime_macros::native_implemented_function;

#[native_implemented_function(insert_element/3)]
pub fn native(
    process: &Process,
    index: Term,
    tuple: Term,
    element: Term,
) -> exception::Result<Term> {
    let initial_inner_tuple: Boxed<Tuple> = tuple.try_into()?;
    let index_one_based: OneBasedIndex = index.try_into().map_err(|_| badarg!())?;

    let length = initial_inner_tuple.len();

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
        .map_err(|error| error.into())
    } else {
        Err(badarg!().into())
    }
}
