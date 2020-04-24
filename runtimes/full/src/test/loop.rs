use std::sync::Arc;

use liblumen_alloc::erts::process::frames;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Atom;

pub fn code(arc_process: &Arc<Process>) -> frames::Result {
    arc_process.reduce();

    Process::call_native_or_yield(arc_process)
}

pub fn function() -> Atom {
    Atom::try_from_str("loop").unwrap()
}

pub fn module() -> Atom {
    Atom::try_from_str("test").unwrap()
}
