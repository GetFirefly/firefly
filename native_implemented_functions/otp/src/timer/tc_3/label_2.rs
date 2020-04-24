use liblumen_alloc::erts::process::{FrameWithArguments, Native};
use liblumen_alloc::erts::term::prelude::*;

use crate::erlang::monotonic_time_0;
use crate::runtime::process::current_process;
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
pub fn frame_with_arguments(before: Term) -> FrameWithArguments {
    assert!(before.is_integer());

    super::label_frame_with_arguments(NATIVE, true, &[before])
}

// Private

const NATIVE: Native = Native::Two(native);

extern "C" fn native(value: Term, before: Term) -> Term {
    let arc_process = current_process();
    arc_process.reduce();

    assert!(before.is_integer());

    arc_process.queue_frame_with_arguments(label_3::frame_with_arguments(before, value));
    arc_process.queue_frame_with_arguments(monotonic_time_0::frame().with_arguments(false, &[]));

    Term::NONE
}
