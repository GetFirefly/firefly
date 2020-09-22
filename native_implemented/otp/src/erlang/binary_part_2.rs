use anyhow::*;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::erlang;

#[native_implemented::function(erlang:binary_part/2)]
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
