use std::sync::Arc;

use anyhow::*;

use liblumen_alloc::erts::exception::AllocResult;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::{code, Process};
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::{Arity, ModuleFunctionArity};

pub fn place_frame_with_arguments(process: &Process, placement: Placement) -> AllocResult<()> {
    process.place_frame(frame(), placement);

    Ok(())
}

// Private

const ARITY: Arity = 0;

fn code(arc_process: &Arc<Process>) -> code::Result {
    arc_process.exit_normal(anyhow!("Out of code").into());

    Ok(())
}

fn frame() -> Frame {
    Frame::new(module_function_arity(), code)
}

fn function() -> Atom {
    Atom::try_from_str("lumen").unwrap()
}

fn module() -> Atom {
    Atom::try_from_str("out_of_code").unwrap()
}

fn module_function_arity() -> Arc<ModuleFunctionArity> {
    Arc::new(ModuleFunctionArity {
        module: module(),
        function: function(),
        arity: ARITY,
    })
}
