mod label_1;
mod label_2;

use std::sync::Arc;

use liblumen_alloc::erts::exception::Alloc;
use liblumen_alloc::erts::process::frames::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::{frames, Process};
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::erts::ModuleFunctionArity;

use liblumen_otp::timer;

pub fn place_frame_with_arguments(
    process: &Process,
    placement: Placement,
    n: Term,
    output: Term,
) -> Result<(), Alloc> {
    assert!(n.is_integer());
    assert!(output.is_boxed_function(), "{:?} is not a function", output);

    process.stack_push(output)?;
    process.stack_push(n)?;
    process.place_frame(frame(), placement);

    Ok(())
}

// Private

/// ```elixir
/// def run(n, output) do
///   {time, value} = :timer.tc(Chain, :create_processes, [n, output])
///   output.("Chain.run(#{n}) in #{time} microseconds")
///   {time, value}
/// end
fn code(arc_process: &Arc<Process>) -> frames::Result {
    arc_process.reduce();

    let n = arc_process.stack_pop().unwrap();
    let output = arc_process.stack_pop().unwrap();

    label_1::place_frame_with_arguments(arc_process, Placement::Replace, output, n).unwrap();

    let module = Atom::str_to_term("Elixir.Chain");
    let function = Atom::str_to_term("create_processes");
    let arguments = arc_process.list_from_slice(&[n, output])?;
    timer::tc_3::place_frame_with_arguments(
        arc_process,
        Placement::Push,
        module,
        function,
        arguments,
    )
    .unwrap();

    Process::call_native_or_yield(arc_process)
}

fn frame() -> Frame {
    Frame::new(module_function_arity(), code)
}

fn function() -> Atom {
    Atom::try_from_str("run").unwrap()
}

fn module_function_arity() -> Arc<ModuleFunctionArity> {
    Arc::new(ModuleFunctionArity {
        module: super::module(),
        function: function(),
        arity: 2,
    })
}
