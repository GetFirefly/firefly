use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::Term;

use crate::erlang::cancel_timer;

#[native_implemented::function(erlang:cancel_timer/1)]
pub fn result(process: &Process, timer_reference: Term) -> Result<Term, NonNull<ErlangException>> {
    cancel_timer(timer_reference, Default::default(), process).map_err(From::from)
}
