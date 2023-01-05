#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;
use std::ptr::NonNull;
use std::sync::Arc;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::Term;

use crate::erlang::start_timer;
use crate::runtime::timer::Format;
use crate::timer;

#[native_implemented::function(erlang:start_timer/4)]
pub fn result(
    arc_process: Arc<Process>,
    time: Term,
    destination: Term,
    message: Term,
    options: Term,
) -> Result<Term, NonNull<ErlangException>> {
    let timer_start_options: timer::start::Options = options.try_into()?;

    start_timer(
        time,
        destination,
        Format::TimeoutTuple,
        message,
        timer_start_options,
        arc_process,
    )
    .map_err(From::from)
}
