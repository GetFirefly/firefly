use std::sync::Arc;

use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::{code, Process};
use liblumen_alloc::erts::term::prelude::*;

use crate::erlang::apply_3;
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
pub fn place_frame_with_arguments(
    process: &Process,
    placement: Placement,
    module: Term,
    function: Term,
    arguments: Term,
) -> code::Result {
    assert!(module.is_atom());
    assert!(function.is_atom());
    assert!(
        arguments.is_list(),
        "arguments ({:?}) are not a list",
        arguments
    );
    process.stack_push(arguments)?;
    process.stack_push(function)?;
    process.stack_push(module)?;
    process.place_frame(frame(process), placement);

    Ok(())
}

// Private

fn code(arc_process: &Arc<Process>) -> code::Result {
    arc_process.reduce();

    let before = arc_process.stack_peek(1).unwrap();
    assert!(before.is_integer());
    let module = arc_process.stack_peek(2).unwrap();
    assert!(module.is_atom(), "module ({:?}) is not an atom", module);
    let function = arc_process.stack_peek(3).unwrap();
    assert!(function.is_atom());
    let arguments = arc_process.stack_peek(4).unwrap();
    assert!(arguments.is_list());

    arc_process.stack_popn(4);

    label_2::place_frame_with_arguments(arc_process, Placement::Replace, before)?;
    apply_3::place_frame_with_arguments(arc_process, Placement::Push, module, function, arguments)?;

    Process::call_code(arc_process)
}

fn frame(process: &Process) -> Frame {
    let module_function_arity = process.current_module_function_arity().unwrap();

    Frame::new(module_function_arity, code)
}
