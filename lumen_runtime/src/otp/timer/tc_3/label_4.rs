use std::sync::Arc;

use liblumen_alloc::erts::exception::Alloc;
use liblumen_alloc::erts::process::code::stack::frame::Placement;
use liblumen_alloc::erts::process::{code, Process};
use liblumen_alloc::erts::term::prelude::*;

use locate_code::locate_code;

use crate::otp::erlang::convert_time_unit_3;

use super::{frame, label_5};

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
    process.place_frame(frame(LOCATION, code), placement);

    Ok(())
}

// Private

#[locate_code]
fn code(arc_process: &Arc<Process>) -> code::Result {
    arc_process.reduce();

    let duration = arc_process.stack_peek(1).unwrap();
    assert!(duration.is_integer());
    let value = arc_process.stack_peek(2).unwrap();

    arc_process.stack_popn(2);

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
