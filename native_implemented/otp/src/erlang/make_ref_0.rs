#[cfg(test)]
mod test;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::runtime::scheduler::SchedulerDependentAlloc;

#[native_implemented::function(make_ref/0)]
pub fn result(process: &Process) -> exception::Result<Term> {
    process.next_reference().map_err(|error| error.into())
}
