mod label_1;
mod label_2;

use std::sync::Arc;

use liblumen_alloc::erts::exception::Alloc;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::code::{self, Code};
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::location::Location;
use liblumen_alloc::Arity;

use locate_code::locate_code;

use lumen_runtime::otp::timer;

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
    process.place_frame(frame(LOCATION, code), placement);

    Ok(())
}

// Private

const ARITY: Arity = 2;

/// ```elixir
/// def run(n, output) do
///   {time, value} = :timer.tc(Chain, :create_processes, [n, output])
///   output.("Chain.run(#{n}) in #{time} microseconds")
///   {time, value}
/// end
#[locate_code]
fn code(arc_process: &Arc<Process>) -> code::Result {
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

    Process::call_code(arc_process)
}

fn frame(location: Location, code: Code) -> Frame {
    Frame::new(super::module(), function(), ARITY, location, code)
}

fn function() -> Atom {
    Atom::try_from_str("run").unwrap()
}
