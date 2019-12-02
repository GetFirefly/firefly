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
use liblumen_alloc::erts::term::prelude::Term;

use lumen_runtime_macros::native_implemented_function;

use crate::otp::erlang::read_timer;
use crate::timer;

#[native_implemented_function(read_timer/2)]
pub fn native(process: &Process, timer_reference: Term, options: Term) -> exception::Result<Term> {
    let read_timer_options: timer::read::Options =
        options.try_into().map_err(|_| badarg!(process))?;

    read_timer(timer_reference, read_timer_options, process)
}
