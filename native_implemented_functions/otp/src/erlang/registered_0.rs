#[cfg(test)]
mod test;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use native_implemented_function::native_implemented_function;

use crate::runtime::registry;

#[native_implemented_function(registered/0)]
pub fn native(process: &Process) -> exception::Result<Term> {
    registry::names(process)
}
