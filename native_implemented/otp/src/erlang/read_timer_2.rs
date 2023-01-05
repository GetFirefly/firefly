#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::Term;

use crate::erlang::read_timer;
use crate::timer;

#[native_implemented::function(erlang:read_timer/2)]
pub fn result(
    process: &Process,
    timer_reference: Term,
    options: Term,
) -> Result<Term, NonNull<ErlangException>> {
    let read_timer_options: timer::read::Options = options.try_into()?;

    read_timer(timer_reference, read_timer_options, process).map_err(From::from)
}
