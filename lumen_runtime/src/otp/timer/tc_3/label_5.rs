use std::sync::Arc;

use liblumen_alloc::erts::exception::Alloc;
use liblumen_alloc::erts::process::code::stack::frame::Placement;
use liblumen_alloc::erts::process::{code, Process};
use liblumen_alloc::erts::term::prelude::*;

use locate_code::locate_code;

use super::frame;

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
    process.place_frame(frame(LOCATION, code), placement);

    Ok(())
}

// Private

#[locate_code]
fn code(arc_process: &Arc<Process>) -> code::Result {
    arc_process.reduce();

    let time = arc_process.stack_peek(1).unwrap();
    assert!(time.is_integer());
    let value = arc_process.stack_peek(2).unwrap();

    let time_value = arc_process.tuple_from_slice(&[time, value])?;
    arc_process.return_from_call(2, time_value)?;

    Process::call_code(arc_process)
}
