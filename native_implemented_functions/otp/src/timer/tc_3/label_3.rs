use liblumen_alloc::erts::process::{FrameWithArguments, Native};
use liblumen_alloc::erts::term::prelude::*;

use crate::erlang::subtract_2;
use crate::runtime::process::current_process;
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
pub fn frame_with_arguments(before: Term, value: Term) -> FrameWithArguments {
    assert!(before.is_integer());
    super::label_frame_with_arguments(NATIVE, true, &[before, value])
}

// Private

const NATIVE: Native = Native::Three(native);

extern "C" fn native(after: Term, before: Term, value: Term) -> Term {
    let arc_process = current_process();
    arc_process.reduce();

    assert!(after.is_integer());
    assert!(before.is_integer());

    arc_process.queue_frame_with_arguments(label_4::frame_with_arguments(value));
    arc_process
        .queue_frame_with_arguments(subtract_2::frame().with_arguments(false, &[after, before]));

    Term::NONE
}
