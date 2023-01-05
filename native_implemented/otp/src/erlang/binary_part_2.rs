use std::ptr::NonNull;

use anyhow::*;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::Term;

use crate::erlang;

#[native_implemented::function(erlang:binary_part/2)]
pub fn result(
    process: &Process,
    binary: Term,
    start_length: Term,
) -> Result<Term, NonNull<ErlangException>> {
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
