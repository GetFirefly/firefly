#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::Term;

use crate::erlang::read_timer;

#[native_implemented::function(erlang:read_timer/1)]
pub fn result(process: &Process, timer_reference: Term) -> Result<Term, NonNull<ErlangException>> {
    read_timer(timer_reference, Default::default(), process).map_err(From::from)
}
