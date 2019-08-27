use std::sync::Arc;

use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::{code, ProcessControlBlock};

use crate::elixir;

/// ```elixir
/// # label: 4
/// # pushed to stack: ()
/// # returned from call: n
/// # full stack: (n)
/// # returns: {time, value}
/// dom(n)
/// ```
pub fn place_frame(process: &ProcessControlBlock, placement: Placement) {
    process.place_frame(frame(process), placement);
}

// Private

fn code(arc_process: &Arc<ProcessControlBlock>) -> code::Result {
    arc_process.reduce();

    let n = arc_process.stack_pop().unwrap();
    assert!(n.is_integer());

    elixir::chain::dom_1::place_frame_with_arguments(arc_process, Placement::Replace, n)?;

    ProcessControlBlock::call_code(arc_process)
}

fn frame(process: &ProcessControlBlock) -> Frame {
    let module_function_arity = process.current_module_function_arity().unwrap();

    Frame::new(module_function_arity, code)
}
