use std::convert::TryInto;
use std::sync::Arc;

use liblumen_alloc::erts::exception::Alloc;
use liblumen_alloc::erts::process::code::stack::frame::Placement;
use liblumen_alloc::erts::process::{code, Process};
use liblumen_alloc::erts::term::prelude::*;

use locate_code::locate_code;

use lumen_runtime::otp::erlang;

use super::{frame, label_2};

///  # label 1
///  # pushed stack: (output)
///  # returned from call: last
///  # full stack: (last, output)
///  # returns: sent
pub fn place_frame_with_arguments(
    process: &Process,
    placement: Placement,
    output: Term,
) -> Result<(), Alloc> {
    process.stack_push(output)?;
    process.place_frame(frame(LOCATION, code), placement);

    Ok(())
}

/// ```elixir
/// # label 1
/// # pushed stack: (output)
/// # returned from call: last
/// # full stack: (last, output)
/// # returns: sent
/// send(last, 0) # start the count by sending a zero to the last process
///
/// # label 2
/// # pushed to stack: (output)
/// # returned from call: sent
/// # full stack: (sent, output)
/// # returns: :ok
/// receive do # and wait for the result to come back to us
///   final_answer when is_integer(final_answer) ->
///     output.("Result is #{inspect(final_answer)}")
///     final_answer
/// end
/// ```
#[locate_code]
fn code(arc_process: &Arc<Process>) -> code::Result {
    // placed on top of stack by return from `elixir::r#enum::reduce_0_code`
    let last = arc_process.stack_pop().unwrap();
    assert!(last.is_local_pid(), "last ({:?}) is not a local pid", last);
    let output = arc_process.stack_pop().unwrap();
    let output_closure: Boxed<Closure> = output.try_into().unwrap();
    assert_eq!(output_closure.arity(), 1);

    // ```elixir
    // # label 2
    // # pushed to stack: (output)
    // # returned from call: sent
    // # full stack: (sent, output)
    // # returns: :ok
    // receive do # and wait for the result to come back to us
    //   final_answer when is_integer(final_answer) ->
    //     output.("Result is #{inspect(final_answer)}")
    // end
    // ```
    label_2::place_frame_with_arguments(arc_process, Placement::Replace, output).unwrap();

    // ```elixir
    // # pushed stack: (last, data)
    // # returned from call: N/A
    // # full stack: (last, data)
    // # returns: sent
    // send(last, 0) # start the count by sending a zero to the last process
    // ```
    let message = arc_process.integer(0).unwrap();
    erlang::send_2::place_frame_with_arguments(arc_process, Placement::Push, last, message)
        .unwrap();

    Process::call_code(arc_process)
}
