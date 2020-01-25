#[cfg(test)]
mod test;

use std::sync::Arc;

use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::{code, Process};
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::Arity;

use locate_code::locate_code;

use crate::elixir::chain::{none_output_1, run_2};

/// ```elixir
/// # pushed to stack: (n)
/// # returned from call: N/A
/// # full stack: (n)
/// # returns: final_answer
/// def none(n) do
///   run(n, &none_output/1)
/// end
/// ```
pub fn place_frame_with_arguments(
    process: &Process,
    placement: Placement,
    n: Term,
) -> code::Result {
    assert!(n.is_integer());
    process.stack_push(n)?;
    process.place_frame(frame(), placement);

    Ok(())
}

// Private

const ARITY: Arity = 1;

#[locate_code]
fn code(arc_process: &Arc<Process>) -> code::Result {
    arc_process.reduce();

    let n = arc_process.stack_pop().unwrap();
    assert!(n.is_integer());

    let none_output_closure = none_output_1::closure(arc_process).unwrap();
    run_2::place_frame_with_arguments(arc_process, Placement::Replace, n, none_output_closure)
        .unwrap();

    Process::call_code(arc_process)
}

fn frame() -> Frame {
    Frame::new(super::module(), function(), ARITY, LOCATION, code)
}

fn function() -> Atom {
    Atom::try_from_str("none").unwrap()
}
