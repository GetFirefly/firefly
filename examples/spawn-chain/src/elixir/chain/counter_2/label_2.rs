use std::convert::TryInto;
use std::sync::Arc;

use liblumen_alloc::erts::process::code::stack::frame::Placement;
use liblumen_alloc::erts::process::{code, Process};
use liblumen_alloc::erts::term::prelude::{Boxed, Closure, Encoded, Term};

use locate_code::locate_code;

use lumen_runtime::otp::erlang;

use super::{frame, label_3};

pub fn place_frame_with_arguments(
    process: &Process,
    placement: Placement,
    next_pid: Term,
    output: Term,
) -> code::Result {
    process.stack_push(output)?;
    process.stack_push(next_pid)?;
    process.place_frame(frame(LOCATION, code), placement);

    Ok(())
}

// Private

/// ```elixir
/// # label 2
/// # pushed to stack: (next_pid, output)
/// # returned from call: sum
/// # full stack: (sum, next_pid, output)
/// # returns: sent
/// sent = send(next_pid, sum)
/// output.("sent #{sent} to #{next_pid}")
/// ```
#[locate_code]
fn code(arc_process: &Arc<Process>) -> code::Result {
    arc_process.reduce();

    let sum = arc_process.stack_pop().unwrap();
    assert!(sum.is_integer());
    let next_pid = arc_process.stack_pop().unwrap();
    assert!(next_pid.is_pid());
    let output = arc_process.stack_pop().unwrap();
    let _: Boxed<Closure> = output.try_into().unwrap();

    // ```elixir
    // # label 3
    // # pushed stack: (output, next_pid)
    // # returned from call: sent
    // # full stack: (sent, output, next_pid)
    // # returns: :ok
    // sent = ...
    // output.("sent #{sent} to #{next_pid}")
    // ```
    label_3::place_frame_with_arguments(arc_process, Placement::Replace, output, next_pid).unwrap();

    // ```elixir
    // # pushed stack: (next_pid, sum)
    // # returned from call: N/A
    // # full stack: (next_pid, sum)
    // # returns: sent
    // send(next_pid, sum)
    erlang::send_2::place_frame_with_arguments(arc_process, Placement::Push, next_pid, sum)
        .unwrap();

    Process::call_code(arc_process)
}
