use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use native_implemented_function::native_implemented_function;

#[native_implemented_function(start/0)]
fn result(process: &Process) -> Term {
    process.wait();

    Term::NONE
}
