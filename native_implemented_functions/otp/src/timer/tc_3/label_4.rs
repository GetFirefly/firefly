use liblumen_alloc::erts::process::{FrameWithArguments, Native};
use liblumen_alloc::erts::term::prelude::*;

use crate::erlang::convert_time_unit_3;
use crate::runtime::process::current_process;
use crate::timer::tc_3::label_5;

/// ```elixir
/// # label 4
/// # pushed to stack: (value)
/// # returned from call: duration
/// # full stack: (duration, value)
/// # returns: time
/// time = :erlang.convert_time_unit(duration, :native, :microsecond)
/// {time, value}
/// ```
pub fn frame_with_arguments(value: Term) -> FrameWithArguments {
    super::label_frame_with_arguments(NATIVE, true, &[value])
}

// Private

const NATIVE: Native = Native::Two(native);

extern "C" fn native(duration: Term, value: Term) -> Term {
    let arc_process = current_process();
    arc_process.reduce();

    assert!(duration.is_integer());

    arc_process.queue_frame_with_arguments(label_5::frame_with_arguments(value));
    arc_process.queue_frame_with_arguments(convert_time_unit_3::frame().with_arguments(
        false,
        &[
            duration,
            Atom::str_to_term("native"),
            Atom::str_to_term("microsecond"),
        ],
    ));

    Term::NONE
}
