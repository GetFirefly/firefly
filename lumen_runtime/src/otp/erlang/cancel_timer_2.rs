// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Term;(

use lumen_runtime_macros::native_implemented_function;

use crate::otp::erlang::cancel_timer;
use crate::timer;

#[native_implemented_function(cancel_timer/2)]
pub fn native(process: &Process, timer_reference: Term, options: Term) -> exception::Result {
    let cancel_timer_options: timer::cancel::Options = options.try_into()?;

    cancel_timer(timer_reference, cancel_timer_options, process)
}
