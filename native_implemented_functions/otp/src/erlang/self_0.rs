#[cfg(test)]
mod test;

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Term;

use native_implemented_function::native_implemented_function;

#[native_implemented_function(self/0)]
pub fn result(process: &Process) -> Term {
    process.pid_term()
}
