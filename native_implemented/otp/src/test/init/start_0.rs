use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

#[native_implemented::function(test:start/0)]
fn result(process: &Process) -> Term {
    process.wait();

    Term::NONE
}
