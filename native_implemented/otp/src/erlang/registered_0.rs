#[cfg(test)]
mod test;

use std::ptr::NonNull;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::Term;

use crate::runtime::registry;

#[native_implemented::function(erlang:registered/0)]
pub fn result(process: &Process) -> Result<Term, NonNull<ErlangException>> {
    registry::names(process)
}
