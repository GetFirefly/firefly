use std::sync::Arc;

use liblumen_alloc::erts::exception::system::Alloc;
use liblumen_alloc::erts::process::code;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::ProcessControlBlock;
use liblumen_alloc::erts::term::{atom_unchecked, Term};

/// ```elixir
/// # label 2
/// # pushed to stack: (value)
/// # returned from call: :ok
/// # full stack: (:ok, value)
/// # returns: value
/// value
pub fn place_frame_with_arguments(
    process: &ProcessControlBlock,
    placement: Placement,
    value: Term,
) -> Result<(), Alloc> {
    assert!(value.is_integer());
    process.stack_push(value)?;
    process.place_frame(frame(process), placement);

    Ok(())
}

// Private

fn code(arc_process: &Arc<ProcessControlBlock>) -> code::Result {
    arc_process.reduce();

    let ok = arc_process.stack_pop().unwrap();
    assert_eq!(ok, atom_unchecked("ok"));
    let value = arc_process.stack_pop().unwrap();

    arc_process.return_from_call(value)?;

    ProcessControlBlock::call_code(arc_process)
}

fn frame(process: &ProcessControlBlock) -> Frame {
    let module_function_arity = process.current_module_function_arity().unwrap();

    Frame::new(module_function_arity, code)
}
