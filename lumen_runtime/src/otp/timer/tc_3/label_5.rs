use std::sync::Arc;

use liblumen_alloc::erts::exception::Alloc;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::{code, Process};
use liblumen_alloc::erts::term::prelude::*;

/// ```elixir
/// # label 5
/// # pushed to stack: (value)
/// # returned from call: time
/// # full stack: (time, value)
/// # returns: {time, value}
/// {time, value}
/// ```
pub fn place_frame_with_arguments(
    process: &Process,
    placement: Placement,
    value: Term,
) -> Result<(), Alloc> {
    process.stack_push(value)?;
    process.place_frame(frame(process), placement);

    Ok(())
}

// Private

fn code(arc_process: &Arc<Process>) -> code::Result {
    arc_process.reduce();

    let time = arc_process.stack_pop().unwrap();
    assert!(time.is_integer());
    let value = arc_process.stack_pop().unwrap();

    let time_value = arc_process.tuple_from_slice(&[time, value])?;
    arc_process.return_from_call(time_value)?;

    Process::call_code(arc_process)
}

fn frame(process: &Process) -> Frame {
    let module_function_arity = process.current_module_function_arity().unwrap();

    Frame::new(module_function_arity, code)
}
