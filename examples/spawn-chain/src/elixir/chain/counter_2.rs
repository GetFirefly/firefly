mod label_1;
mod label_2;
mod label_3;

use std::convert::TryInto;
use std::sync::Arc;

use liblumen_alloc::erts::process::frames;
use liblumen_alloc::erts::process::frames::stack::frame::Placement;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::Arity;

pub fn export() {
    lumen_rt_full::code::export::insert(super::module(), function(), ARITY, code);
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
fn code(arc_process: &Arc<Process>) -> frames::Result {
    arc_process.reduce();

    let next_pid = arc_process.stack_pop().unwrap();
    assert!(next_pid.is_pid());
    let output = arc_process.stack_pop().unwrap();
    // is_function(output, ...)
    let output_closure: Boxed<Closure> = output.try_into().unwrap();

    // is_function(output, 1)
    let output_module_function_arity = output_closure.module_function_arity();
    assert_eq!(output_module_function_arity.arity, 1);

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

    Process::call_native_or_yield(arc_process)
}

fn function() -> Atom {
    Atom::try_from_str("counter").unwrap()
}
