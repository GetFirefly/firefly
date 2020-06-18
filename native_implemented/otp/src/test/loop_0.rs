use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

pub use super::module;

#[native_implemented::function(test:loop/0)]
fn result(process: &Process) -> Term {
    process.wait();
    process.queue_frame_with_arguments(frame().with_arguments(false, &[]));

    Term::NONE
}
