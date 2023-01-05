#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;
use std::ptr::NonNull;

use anyhow::*;

use firefly_number::TryIntoIntegerError;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::Term;

use crate::binary;

/// The one-based indexing for binaries used by this function is deprecated. New code is to use
/// [crate::binary::bin_to_list] instead. All functions in module [crate::binary]
/// consistently use zero-based indexing.
#[native_implemented::function(erlang:binary_to_list/3)]
pub fn result(
    process: &Process,
    binary: Term,
    start: Term,
    stop: Term,
) -> Result<Term, NonNull<ErlangException>> {
    let one_based_start_usize: usize = try_into_one_based("start", start)?;
    let one_based_stop_usize: usize = try_into_one_based("stop", stop)?;

    if one_based_start_usize <= one_based_stop_usize {
        let zero_based_start_usize = one_based_start_usize - 1;
        let zero_based_stop_usize = one_based_stop_usize - 1;

        let length_usize = zero_based_stop_usize - zero_based_start_usize + 1;

        binary::bin_to_list(
            binary,
            process.integer(zero_based_start_usize),
            process.integer(length_usize).unwrap(),
            process,
        )
    } else {
        Err(TryIntoIntegerError::OutOfRange)
            .context(format!(
                "start ({}) must be less than or equal to stop ({})",
                start, stop
            ))
            .map_err(From::from)
    }
}

fn one_based_context(name: &str, value: Term) -> String {
    format!(
        "{} ({}) must be a one-based integer index between 1 and the byte size of the binary",
        name, value
    )
}

fn try_into_one_based(name: &str, value_term: Term) -> InternalResult<usize> {
    let value_usize: usize = value_term
        .try_into()
        .with_context(|| one_based_context(name, value_term))?;

    if 1 <= value_usize {
        Ok(value_usize)
    } else {
        Err(TryIntoIntegerError::OutOfRange)
            .context(one_based_context(name, value_term))
            .map_err(From::from)
    }
}
