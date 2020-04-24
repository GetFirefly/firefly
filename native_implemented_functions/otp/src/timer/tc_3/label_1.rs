use liblumen_alloc::erts::process::{FrameWithArguments, Native};
use liblumen_alloc::erts::term::prelude::*;

use crate::erlang::apply_3;
use crate::runtime::process::current_process;
use crate::timer::tc_3::label_2;

/// ```elixir
/// # label 1
/// # pushed to stack: (module, function arguments, before)
/// # returned from call: before
/// # full stack: (before, module, function arguments)
/// # returns: value
/// value = apply(module, function, arguments)
/// after = :erlang.monotonic_time()
/// duration = after - before
/// time = :erlang.convert_time_unit(duration, :native, :microsecond)
/// {time, value}
/// ```
pub fn frame_with_arguments(module: Term, function: Term, arguments: Term) -> FrameWithArguments {
    assert!(module.is_atom());
    assert!(function.is_atom());
    assert!(
        arguments.is_list(),
        "arguments ({:?}) are not a list",
        arguments
    );

    super::label_frame_with_arguments(NATIVE, true, &[module, function, arguments])
}

// Private

const NATIVE: Native = Native::Four(native);

extern "C" fn native(before: Term, module: Term, function: Term, arguments: Term) -> Term {
    let arc_process = current_process();
    arc_process.reduce();

    assert!(before.is_integer());
    assert!(module.is_atom(), "module ({:?}) is not an atom", module);
    assert!(function.is_atom());
    assert!(arguments.is_list());

    arc_process.queue_frame_with_arguments(label_2::frame_with_arguments(before));
    arc_process
        .queue_frame_with_arguments(apply_3::frame_with_arguments(module, function, arguments));

    Term::NONE
}
