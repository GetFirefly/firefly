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
use liblumen_alloc::erts::term::prelude::Term;(

use lumen_runtime_macros::native_implemented_function;

use crate::otp;

/// The one-based indexing for binaries used by this function is deprecated. New code is to use
/// [crate::otp::binary::bin_to_list] instead. All functions in module [crate::otp::binary]
/// consistently use zero-based indexing.
#[native_implemented_function(binary_to_list/3)]
pub fn native(process: &Process, binary: Term, start: Term, stop: Term) -> exception::Result {
    let one_based_start_usize: usize = start.try_into()?;

    if 1 <= one_based_start_usize {
        let one_based_stop_usize: usize = stop.try_into()?;

        if one_based_start_usize <= one_based_stop_usize {
            let zero_based_start_usize = one_based_start_usize - 1;
            let zero_based_stop_usize = one_based_stop_usize - 1;

            let length_usize = zero_based_stop_usize - zero_based_start_usize + 1;

            otp::binary::bin_to_list(
                binary,
                process.integer(zero_based_start_usize)?,
                process.integer(length_usize)?,
                process,
            )
        } else {
            Err(badarg!().into())
        }
    } else {
        Err(badarg!().into())
    }
}
