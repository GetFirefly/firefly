#[cfg(test)]
mod test;

use std::sync::Arc;

use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::{code, ProcessControlBlock};
use liblumen_alloc::erts::term::{Atom, Term};
use liblumen_alloc::ModuleFunctionArity;

pub fn place_frame(process: &ProcessControlBlock, placement: Placement) {
    process.place_frame(frame(), placement);
}

// Private

pub(crate) fn code(arc_process: &Arc<ProcessControlBlock>) -> code::Result {
    arc_process.reduce();

    let self_pid = native(arc_process);
    arc_process.return_from_call(self_pid)?;

    ProcessControlBlock::call_code(arc_process)
}

fn frame() -> Frame {
    Frame::new(module_function_arity(), code)
}

fn function() -> Atom {
    Atom::try_from_str("self").unwrap()
}

fn module_function_arity() -> Arc<ModuleFunctionArity> {
    Arc::new(ModuleFunctionArity {
        module: super::module(),
        function: function(),
        arity: 0,
    })
}

fn native(process_control_block: &ProcessControlBlock) -> Term {
    process_control_block.pid_term()
}
