use std::sync::Arc;

use liblumen_alloc::erts::exception::system::Alloc;
use liblumen_alloc::erts::process::code;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::ProcessControlBlock;
use liblumen_alloc::erts::term::{atom_unchecked, Term};

/// ```elixir
/// # label 2
/// # pushed to stack: ({time, value})
/// # returned from call: :ok
/// # full stack: (:ok, {time, value})
/// # returns: {time, value}
/// {time, value}
pub fn place_frame_with_arguments(
    process: &ProcessControlBlock,
    placement: Placement,
    time_value: Term,
) -> Result<(), Alloc> {
    assert!(time_value.is_tuple());
    process.stack_push(time_value)?;
    process.place_frame(frame(process), placement);

    Ok(())
}

// Private

fn code(arc_process: &Arc<ProcessControlBlock>) -> code::Result {
    arc_process.reduce();

    let ok = arc_process.stack_pop().unwrap();
    assert_eq!(ok, atom_unchecked("ok"));
    let time_value = arc_process.stack_pop().unwrap();

    arc_process.return_from_call(time_value)?;

    ProcessControlBlock::call_code(arc_process)
}

fn frame(process: &ProcessControlBlock) -> Frame {
    let module_function_arity = process.current_module_function_arity().unwrap();

    Frame::new(module_function_arity, code)
}
