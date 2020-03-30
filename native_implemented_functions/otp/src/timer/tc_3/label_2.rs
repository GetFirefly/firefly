use std::sync::Arc;

use liblumen_alloc::erts::exception::Alloc;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::{code, Process};
use liblumen_alloc::erts::term::prelude::*;

use crate::erlang::monotonic_time_0;
use crate::timer::tc_3::label_3;

/// ```elixir
/// # label 2
/// # pushed to stack: (before)
/// # returned from call: value
/// # full stack: (value, before)
/// # returns: after
/// after = :erlang.monotonic_time()
/// duration = after - before
/// time = :erlang.convert_time_unit(duration, :native, :microsecond)
/// {time, value}
/// ```
pub fn place_frame_with_arguments(
    process: &Process,
    placement: Placement,
    before: Term,
) -> Result<(), Alloc> {
    assert!(before.is_integer());
    process.stack_push(before)?;
    process.place_frame(frame(process), placement);

    Ok(())
}

// Private

fn code(arc_process: &Arc<Process>) -> code::Result {
    arc_process.reduce();

    let value = arc_process.stack_pop().unwrap();
    let before = arc_process.stack_pop().unwrap();
    assert!(before.is_integer());

    label_3::place_frame_with_arguments(arc_process, Placement::Replace, before, value)?;
    monotonic_time_0::place_frame_with_arguments(arc_process, Placement::Push)?;

    Process::call_code(arc_process)
}

fn frame(process: &Process) -> Frame {
    let module_function_arity = process.current_module_function_arity().unwrap();

    Frame::new(module_function_arity, code)
}
