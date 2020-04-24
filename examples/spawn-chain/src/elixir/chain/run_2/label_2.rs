use std::sync::Arc;

use liblumen_alloc::erts::exception::Alloc;
use liblumen_alloc::erts::process::frames;
use liblumen_alloc::erts::process::frames::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

/// ```elixir
/// # label 2
/// # pushed to stack: ({time, value})
/// # returned from call: :ok
/// # full stack: (:ok, {time, value})
/// # returns: {time, value}
/// {time, value}
pub fn place_frame_with_arguments(
    process: &Process,
    placement: Placement,
    time_value: Term,
) -> Result<(), Alloc> {
    assert!(time_value.is_boxed_tuple());
    process.stack_push(time_value).unwrap();
    process.place_frame(frame(process), placement);

    Ok(())
}

// Private

fn code(arc_process: &Arc<Process>) -> frames::Result {
    arc_process.reduce();

    let ok = arc_process.stack_peek(1).unwrap();
    assert_eq!(ok, Atom::str_to_term("ok"));
    let time_value = arc_process.stack_peek(2).unwrap();

    arc_process.return_from_call(2, time_value).unwrap();

    Process::call_native_or_yield(arc_process)
}

fn frame(process: &Process) -> Frame {
    let module_function_arity = process.current_module_function_arity().unwrap();

    Frame::new(module_function_arity, code)
}
