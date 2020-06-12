#[cfg(all(not(any(target_arch = "wasm32", feature = "runtime_minimal")), test))]
mod test;

use std::convert::TryInto;
use std::sync::Arc;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Term;

use crate::erlang::start_timer;
use crate::runtime::timer::Timeout;
use crate::timer;

#[native_implemented::function(send_after/4)]
pub fn result(
    arc_process: Arc<Process>,
    time: Term,
    destination: Term,
    message: Term,
    options: Term,
) -> exception::Result<Term> {
    let timer_start_options: timer::start::Options = options.try_into()?;

    start_timer(
        time,
        destination,
        Timeout::Message,
        message,
        timer_start_options,
        arc_process,
    )
    .map_err(From::from)
}
