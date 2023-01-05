#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::sync::Arc;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::Term;

use crate::erlang::start_timer;
use crate::runtime::timer::Format;

#[native_implemented::function(erlang:send_after/3)]
pub fn result(
    arc_process: Arc<Process>,
    time: Term,
    destination: Term,
    message: Term,
) -> Result<Term, NonNull<ErlangException>> {
    start_timer(
        time,
        destination,
        Format::Message,
        message,
        Default::default(),
        arc_process,
    )
    .map_err(From::from)
}
