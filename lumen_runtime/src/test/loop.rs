use std::sync::Arc;

use liblumen_alloc::erts::process::code;
use liblumen_alloc::erts::process::ProcessControlBlock;
use liblumen_alloc::erts::term::Atom;

pub fn code(arc_process: &Arc<ProcessControlBlock>) -> code::Result {
    arc_process.reduce();

    ProcessControlBlock::call_code(arc_process)
}

pub fn function() -> Atom {
    Atom::try_from_str("loop").unwrap()
}

pub fn module() -> Atom {
    Atom::try_from_str("test").unwrap()
}
