#[cfg(test)]
mod test;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Term;

use lumen_runtime_macros::native_implemented_function;

use crate::erlang::unique_integer::unique_integer;

#[native_implemented_function(unique_integer/0)]
pub fn native(process: &Process) -> exception::Result<Term> {
    unique_integer(process, Default::default())
}
