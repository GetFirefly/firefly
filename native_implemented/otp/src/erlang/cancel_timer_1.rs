use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Term;

use crate::erlang::cancel_timer;

#[native_implemented::function(erlang:cancel_timer/1)]
pub fn result(process: &Process, timer_reference: Term) -> exception::Result<Term> {
    cancel_timer(timer_reference, Default::default(), process).map_err(From::from)
}
