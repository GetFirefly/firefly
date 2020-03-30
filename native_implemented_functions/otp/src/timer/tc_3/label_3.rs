use std::sync::Arc;

use liblumen_alloc::erts::exception::Alloc;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::{code, Process};
use liblumen_alloc::erts::term::prelude::*;

use crate::erlang::subtract_2;
use crate::timer::tc_3::label_4;

/// ```elixir
/// # label 3
/// # pushed to stack: (before, value)
/// # returned from call: after
/// # full stack: (after, before, value)
/// # returns: duration
/// duration = after - before
/// time = :erlang.convert_time_unit(duration, :native, :microsecond)
/// {time, value}
/// ```
pub fn place_frame_with_arguments(
    process: &Process,
    placement: Placement,
    before: Term,
    value: Term,
) -> Result<(), Alloc> {
    assert!(before.is_integer());
    process.stack_push(value)?;
    process.stack_push(before)?;
    process.place_frame(frame(process), placement);

    Ok(())
}

// Private

fn code(arc_process: &Arc<Process>) -> code::Result {
    arc_process.reduce();

    let after = arc_process.stack_peek(1).unwrap();
    assert!(after.is_integer());
    let before = arc_process.stack_peek(2).unwrap();
    assert!(before.is_integer());
    let value = arc_process.stack_peek(3).unwrap();

    arc_process.stack_popn(3);

    label_4::place_frame_with_arguments(arc_process, Placement::Replace, value)?;
    subtract_2::place_frame_with_arguments(arc_process, Placement::Push, after, before)?;
    Process::call_code(arc_process)
}

fn frame(process: &Process) -> Frame {
    let module_function_arity = process.current_module_function_arity().unwrap();

    Frame::new(module_function_arity, code)
}
