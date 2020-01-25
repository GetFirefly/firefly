use std::sync::Arc;

use liblumen_alloc::erts::exception::Alloc;
use liblumen_alloc::erts::process::code::stack::frame::Placement;
use liblumen_alloc::erts::process::{code, Process};
use liblumen_alloc::erts::term::prelude::*;

use locate_code::locate_code;

use crate::otp::erlang::monotonic_time_0;

use super::{frame, label_3};

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
    process.place_frame(frame(LOCATION, code), placement);

    Ok(())
}

// Private

#[locate_code]
fn code(arc_process: &Arc<Process>) -> code::Result {
    arc_process.reduce();

    let value = arc_process.stack_pop().unwrap();
    let before = arc_process.stack_pop().unwrap();
    assert!(before.is_integer());

    label_3::place_frame_with_arguments(arc_process, Placement::Replace, before, value)?;
    monotonic_time_0::place_frame_with_arguments(arc_process, Placement::Push)?;

    Process::call_code(arc_process)
}
