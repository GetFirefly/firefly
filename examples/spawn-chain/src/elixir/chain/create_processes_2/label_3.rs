use std::sync::Arc;

use liblumen_alloc::erts::exception::Alloc;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::{code, Process};
use liblumen_alloc::erts::term::prelude::*;

/// ```elixir
/// # pushed to stack: (final_answer)
/// # returned from call: :ok
/// # full stack: (:ok, final_answer)
/// # returns: final_answer
/// final_answer
/// ```
pub fn place_frame_with_arguments(
    process: &Process,
    placement: Placement,
    final_answer: Term,
) -> Result<(), Alloc> {
    process.stack_push(final_answer)?;
    process.place_frame(frame(process), placement);

    Ok(())
}

fn code(arc_process: &Arc<Process>) -> code::Result {
    arc_process.reduce();

    let ok = arc_process.stack_pop().unwrap();
    assert_eq!(ok, Atom::str_to_term("ok"));
    let final_answer = arc_process.stack_pop().unwrap();
    assert!(final_answer.is_integer());

    arc_process.return_from_call(final_answer)?;

    Process::call_code(arc_process)
}

fn frame(process: &Process) -> Frame {
    let module_function_arity = process.current_module_function_arity().unwrap();

    Frame::new(module_function_arity, code)
}
