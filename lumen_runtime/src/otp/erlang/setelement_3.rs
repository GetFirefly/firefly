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
use liblumen_alloc::erts::term::index::ZeroBasedIndex;
use liblumen_alloc::erts::term::prelude::*;

use lumen_runtime_macros::native_implemented_function;

#[native_implemented_function(setelement/3)]
pub fn native(process: &Process, index: Term, tuple: Term, value: Term) -> exception::Result<Term> {
    let initial_inner_tuple: Boxed<Tuple> = tuple.try_into()?;
    let index_zero_based: ZeroBasedIndex = index.try_into()?;

    let length = initial_inner_tuple.len();

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
        Err(badarg!().into())
    }
}
