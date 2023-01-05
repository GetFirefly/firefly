use std::convert::TryInto;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::Term;

use crate::erlang::cancel_timer;
use crate::timer;

#[native_implemented::function(erlang:cancel_timer/2)]
pub fn result(
    process: &Process,
    timer_reference: Term,
    options: Term,
) -> Result<Term, NonNull<ErlangException>> {
    let cancel_timer_options: timer::cancel::Options = options.try_into()?;

    cancel_timer(timer_reference, cancel_timer_options, process).map_err(From::from)
}
