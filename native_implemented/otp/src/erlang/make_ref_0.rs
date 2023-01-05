#[cfg(test)]
mod test;

use std::ptr::NonNull;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::Term;

use crate::runtime::scheduler::next_local_reference_term;

#[native_implemented::function(erlang:make_ref/0)]
pub fn result(process: &Process) -> Result<Term, NonNull<ErlangException>> {
    next_local_reference_term(process).map_err(ErlangException::from)
}
