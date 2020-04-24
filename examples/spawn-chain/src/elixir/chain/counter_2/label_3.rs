use std::convert::TryInto;
use std::sync::Arc;

use liblumen_alloc::erts::exception::Alloc;
use liblumen_alloc::erts::process::frames::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::{frames, Process};
use liblumen_alloc::erts::term::prelude::*;

pub fn place_frame_with_arguments(
    process: &Process,
    placement: Placement,
    output: Term,
    next_pid: Term,
) -> Result<(), Alloc> {
    process.stack_push(next_pid).unwrap();
    process.stack_push(output).unwrap();
    process.place_frame(frame(process), placement);

    Ok(())
}

/// ```elixir
/// # label 3
/// # pushed stack: (output, next_pid)
/// # returned from call: sent
/// # full stack: (sent, output, next_pid)
/// # returns: :ok
/// sent = ...
/// output.("sent #{sent} to #{next_pid}")
/// ```
fn code(arc_process: &Arc<Process>) -> frames::Result {
    arc_process.reduce();

    let sent = arc_process.stack_pop().unwrap();
    assert!(sent.is_integer());
    let output = arc_process.stack_pop().unwrap();
    assert!(output.is_boxed_function());
    let next_pid = arc_process.stack_pop().unwrap();
    assert!(next_pid.is_pid());

    let output_closure: Boxed<Closure> = output.try_into().unwrap();
    assert_eq!(output_closure.arity(), 1);

    // TODO use `<>` and `to_string` instead of `format!` to properly emulate interpolation
    let data = arc_process
        .binary_from_str(&format!("sent {} to {}", sent, next_pid))
        .unwrap();
    output_closure
        .place_frame_with_arguments(arc_process, Placement::Replace, vec![data])
        .unwrap();

    Process::call_native_or_yield(arc_process)
}

fn frame(process: &Process) -> Frame {
    let module_function_arity = process.current_module_function_arity().unwrap();

    Frame::new(module_function_arity, code)
}
