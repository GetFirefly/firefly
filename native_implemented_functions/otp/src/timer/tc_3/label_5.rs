use liblumen_alloc::erts::process::{FrameWithArguments, Native};
use liblumen_alloc::erts::term::prelude::*;

use crate::runtime::process::current_process;

/// ```elixir
/// # label 5
/// # pushed to stack: (value)
/// # returned from call: time
/// # full stack: (time, value)
/// # returns: {time, value}
/// {time, value}
/// ```
pub fn frame_with_arguments(value: Term) -> FrameWithArguments {
    super::label_frame_with_arguments(NATIVE, true, &[value])
}

// Private

const NATIVE: Native = Native::Two(native);

extern "C" fn native(time: Term, value: Term) -> Term {
    let arc_process = current_process();
    arc_process.reduce();

    assert!(time.is_integer());

    let exception_result = arc_process
        .tuple_from_slice(&[time, value])
        .map_err(From::from);

    arc_process.return_status(exception_result)
}
