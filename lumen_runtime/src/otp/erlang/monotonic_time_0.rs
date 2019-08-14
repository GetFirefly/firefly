#[cfg(test)]
mod test;

use std::sync::Arc;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::code::{self, result_from_exception};
use liblumen_alloc::erts::process::ProcessControlBlock;
use liblumen_alloc::erts::term::Atom;
use liblumen_alloc::ModuleFunctionArity;

use crate::time::{monotonic, Unit::Native};

pub fn native(process_control_block: &ProcessControlBlock) -> exception::Result {
    let big_int = monotonic::time(Native);

    Ok(process_control_block.integer(big_int)?)
}

pub fn place_frame(process: &ProcessControlBlock, placement: Placement) {
    process.place_frame(frame(), placement);
}

// Private

fn code(arc_process: &Arc<ProcessControlBlock>) -> code::Result {
    arc_process.reduce();

    match native(arc_process) {
        Ok(time) => {
            arc_process.return_from_call(time)?;

            ProcessControlBlock::call_code(arc_process)
        }
        Err(exception) => result_from_exception(arc_process, exception),
    }
}

fn frame() -> Frame {
    Frame::new(module_function_arity(), code)
}

fn function() -> Atom {
    Atom::try_from_str("monotonic_time").unwrap()
}

fn module_function_arity() -> Arc<ModuleFunctionArity> {
    Arc::new(ModuleFunctionArity {
        module: super::module(),
        function: function(),
        arity: 0,
    })
}
