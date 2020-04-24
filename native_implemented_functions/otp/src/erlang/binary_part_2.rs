// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use anyhow::*;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use native_implemented_function::native_implemented_function;

use crate::erlang;

#[native_implemented_function(binary_part/2)]
pub fn result(process: &Process, binary: Term, start_length: Term) -> exception::Result<Term> {
    let start_length_tuple = term_try_into_tuple!(start_length)?;

    if start_length_tuple.len() == 2 {
        erlang::binary_part_3::result(
            process,
            binary,
            start_length_tuple[0],
            start_length_tuple[1],
        )
    } else {
        Err(anyhow!(
            "start_length ({}) is a tuple, but not 2-arity",
            start_length
        )
        .into())
    }
}
