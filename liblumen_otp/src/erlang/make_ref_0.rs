#[cfg(test)]
mod test;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use lumen_runtime_macros::native_implemented_function;

use lumen_runtime::process::SchedulerDependentAlloc;

#[native_implemented_function(make_ref/0)]
pub fn native(process: &Process) -> exception::Result<Term> {
    process.next_reference().map_err(|error| error.into())
}
