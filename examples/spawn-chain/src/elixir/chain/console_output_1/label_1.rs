use std::sync::Arc;
use std::convert::TryInto;

use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::{code, Process};
use liblumen_alloc::erts::term::prelude::{Term, Pid};

use crate::elixir;

pub fn place_frame_with_arguments(
    process: &Process,
    placement: Placement,
    text: Term,
) -> code::Result {
    process.stack_push(text)?;
    process.place_frame(frame(process), placement);

    Ok(())
}

// Private

/// ```elixir
/// # label 1
/// # pushed to stack: (text)
/// # returned from call: self
/// # full stack: (self, text)
/// # returns: :ok
/// IO.puts("#{self()} #{text}")
/// ```
fn code(arc_process: &Arc<Process>) -> code::Result {
    arc_process.reduce();

    let self_pid: Pid = arc_process.stack_pop()
        .unwrap()
        .try_into()
        .unwrap();

    let text = arc_process.stack_pop().unwrap();

    // TODO use `<>` and `to_string` to emulate interpolation properly
    let full_text = arc_process
        .binary_from_str(&format!("pid={} {}", self_pid, text))
        .unwrap();
    elixir::io::puts_1::place_frame_with_arguments(arc_process, Placement::Replace, full_text)
        .unwrap();

    Process::call_code(arc_process)
}

fn frame(process: &Process) -> Frame {
    let module_function_arity = process.current_module_function_arity().unwrap();

    Frame::new(module_function_arity, code)
}
