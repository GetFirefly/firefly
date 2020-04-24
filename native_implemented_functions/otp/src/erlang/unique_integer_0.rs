#[cfg(test)]
mod test;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Term;

use native_implemented_function::native_implemented_function;

use crate::erlang::unique_integer::unique_integer;

#[native_implemented_function(unique_integer/0)]
pub fn result(process: &Process) -> exception::Result<Term> {
    unique_integer(process, Default::default())
}
