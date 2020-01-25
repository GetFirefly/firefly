use std::sync::Arc;

use liblumen_alloc::erts::process::code::stack::frame::Placement;
use liblumen_alloc::erts::process::{code, Process};
use liblumen_alloc::erts::term::prelude::Encoded;

use locate_code::locate_code;

use lumen_runtime::otp::erlang;

use super::{frame, label_4};

/// ```elixir
/// # label: 3
/// # pushed to stack: ()
/// # returned from call: value_string
/// # full stack: (value_string)
/// # returns: n
/// n = :erlang.binary_to_integer(value_string)
/// dom(n)
/// ```
pub fn place_frame(process: &Process, placement: Placement) {
    process.place_frame(frame(LOCATION, code), placement);
}

// Private

#[locate_code]
fn code(arc_process: &Arc<Process>) -> code::Result {
    arc_process.reduce();

    let value_string = arc_process.stack_pop().unwrap();
    assert!(value_string.is_binary());

    // ```elixir
    // # label: 4
    // # pushed to stack: ()
    // # returned from call: n
    // # full stack: (n)
    // # returns: {time, value}
    // dom(n)
    // ```
    label_4::place_frame(arc_process, Placement::Replace);

    erlang::binary_to_integer_1::place_frame_with_arguments(
        arc_process,
        Placement::Push,
        value_string,
    )
    .unwrap();

    Process::call_code(arc_process)
}
