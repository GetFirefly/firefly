#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Term;

use crate::erlang::cancel_timer;
use crate::timer;

#[native_implemented::function(erlang:cancel_timer/2)]
pub fn result(process: &Process, timer_reference: Term, options: Term) -> exception::Result<Term> {
    let cancel_timer_options: timer::cancel::Options = options.try_into()?;

    cancel_timer(timer_reference, cancel_timer_options, process).map_err(From::from)
}
