mod label_1;
mod label_2;
mod label_3;

use std::convert::TryInto;
use std::sync::Arc;

use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::code::{self, Code};
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::closure::Definition;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::location::Location;
use liblumen_alloc::Arity;

use locate_code::locate_code;

pub fn export() {
    let definition = Definition::Export {
        function: function(),
    };
    lumen_runtime::code::insert(super::module(), definition, ARITY, LOCATED_CODE);
}

// Private

const ARITY: Arity = 2;

/// ```elixir
/// def counter(next_pid, output) when is_function(output, 1) do
///   output.("spawned")
///
///   receive do
///     n ->
///       output.("received #{n}")
///       sent = send(next_pid, n + 1)
///       output.("sent #{sent} to #{next_pid}")
///   end
/// end
/// ```
#[locate_code]
fn code(arc_process: &Arc<Process>) -> code::Result {
    arc_process.reduce();

    let next_pid = arc_process.stack_pop().unwrap();
    assert!(next_pid.is_pid());
    let output = arc_process.stack_pop().unwrap();
    // is_function(output, ...)
    let output_closure: Boxed<Closure> = output.try_into().unwrap();

    // is_function(output, 1)
    assert_eq!(output_closure.arity(), 1);

    // # label 1
    // # pushed to stack: (next_pid, output)
    // # returned from called: :ok
    // # full stack: (:ok, next_pid, output)
    // receive do
    //   n ->
    //     output.("received #{n}")
    //     sent = send(next_pid, n + 1)
    //     output.("sent #{sent} to #{next_pid}")
    // end
    label_1::place_frame_with_arguments(arc_process, Placement::Replace, next_pid, output).unwrap();

    // ```elixir
    // output.("spawned")
    // ```
    let output_data = arc_process.binary_from_str("spawned").unwrap();
    output_closure
        .place_frame_with_arguments(arc_process, Placement::Push, vec![output_data])
        .unwrap();

    Process::call_code(arc_process)
}

fn frame(location: Location, code: Code) -> Frame {
    Frame::new(super::module(), function(), ARITY, location, code)
}

fn function() -> Atom {
    Atom::try_from_str("counter").unwrap()
}
