use std::convert::TryInto;
use std::sync::Arc;

use liblumen_alloc::erts::exception::system::Alloc;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::{code, ProcessControlBlock};
use liblumen_alloc::erts::term::{Boxed, Closure, Term};

use lumen_runtime::otp::erlang;

use crate::elixir::chain::create_processes_2::label_2;

///  # label 1
///  # pushed stack: (output)
///  # returned from call: last
///  # full stack: (last, output)
///  # returns: sent
pub fn place_frame_with_arguments(
    process: &ProcessControlBlock,
    placement: Placement,
    output: Term,
) -> Result<(), Alloc> {
    process.stack_push(output)?;
    process.place_frame(frame(process), placement);

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
fn code(arc_process: &Arc<ProcessControlBlock>) -> code::Result {
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
    label_2::place_frame_with_arguments(arc_process, Placement::Replace, output)?;

    // ```elixir
    // # pushed stack: (last, data)
    // # returned from call: N/A
    // # full stack: (last, data)
    // # returns: sent
    // send(last, 0) # start the count by sending a zero to the last process
    // ```
    let message = arc_process.integer(0)?;
    erlang::send_2::place_frame_with_arguments(arc_process, Placement::Push, last, message)?;

    ProcessControlBlock::call_code(arc_process)
}

fn frame(process: &ProcessControlBlock) -> Frame {
    let module_function_arity = process.current_module_function_arity().unwrap();

    Frame::new(module_function_arity, code)
}
