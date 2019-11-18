#[cfg(test)]
mod test;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use lumen_runtime_macros::native_implemented_function;

use crate::registry;

#[native_implemented_function(registered/0)]
pub fn native(process: &Process) -> exception::Result<Term> {
    registry::names(process)
}
