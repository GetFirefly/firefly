#[cfg(test)]
mod test;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use native_implemented_function::native_implemented_function;

use crate::runtime::scheduler::SchedulerDependentAlloc;

#[native_implemented_function(make_ref/0)]
pub fn result(process: &Process) -> exception::Result<Term> {
    process.next_reference().map_err(|error| error.into())
}
