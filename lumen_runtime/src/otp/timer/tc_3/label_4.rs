use std::sync::Arc;

use liblumen_alloc::erts::exception::Alloc;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::{code, Process};
use liblumen_alloc::erts::term::prelude::*;

use crate::otp::erlang::convert_time_unit_3;
use crate::otp::timer::tc_3::label_5;

/// ```elixir
/// # label 4
/// # pushed to stack: (value)
/// # returned from call: duration
/// # full stack: (duration, value)
/// # returns: time
/// time = :erlang.convert_time_unit(duration, :native, :microsecond)
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

    let duration = arc_process.stack_pop().unwrap();
    assert!(duration.is_integer());
    let value = arc_process.stack_pop().unwrap();

    label_5::place_frame_with_arguments(arc_process, Placement::Replace, value)?;
    convert_time_unit_3::place_frame_with_arguments(
        arc_process,
        Placement::Push,
        duration,
        Atom::str_to_term("native"),
        Atom::str_to_term("microsecond"),
    )?;

    Process::call_code(arc_process)
}

fn frame(process: &Process) -> Frame {
    let module_function_arity = process.current_module_function_arity().unwrap();

    Frame::new(module_function_arity, code)
}
