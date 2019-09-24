#[cfg(test)]
mod test;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;

use lumen_runtime_macros::native_implemented_function;

use crate::process::SchedulerDependentAlloc;

#[native_implemented_function(make_ref/0)]
pub fn native(process: &Process) -> exception::Result {
    process.next_reference().map_err(|error| error.into())
}
