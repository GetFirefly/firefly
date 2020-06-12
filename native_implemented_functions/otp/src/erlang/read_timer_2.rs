#[cfg(all(not(any(target_arch = "wasm32", feature = "runtime_minimal")), test))]
mod test;

use std::convert::TryInto;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Term;

use crate::erlang::read_timer;
use crate::timer;

#[native_implemented::function(read_timer/2)]
pub fn result(process: &Process, timer_reference: Term, options: Term) -> exception::Result<Term> {
    let read_timer_options: timer::read::Options = options.try_into()?;

    read_timer(timer_reference, read_timer_options, process).map_err(From::from)
}
