mod label_1;
mod label_2;
mod label_3;
mod label_4;
mod label_5;

use std::sync::Arc;

use liblumen_alloc::erts::exception::Alloc;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::code::Code;
use liblumen_alloc::erts::process::{code, Process};
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::location::Location;
use liblumen_alloc::Arity;

use locate_code::locate_code;

use crate::otp::erlang::monotonic_time_0;

use super::module;

pub fn place_frame_with_arguments(
    process: &Process,
    placement: Placement,
    module: Term,
    function: Term,
    arguments: Term,
) -> Result<(), Alloc> {
    process.stack_push(arguments)?;
    process.stack_push(function)?;
    process.stack_push(module)?;
    process.place_frame(frame(LOCATION, code), placement);

    Ok(())
}

// Private

const ARITY: Arity = 3;

/// ```elixir
/// def tc(module, function, arguments) do
///   before = :erlang.monotonic_time()
///   value = apply(module, function, arguments)
///   after = :erlang.monotonic_time()
///   duration = after - before
///   time = :erlang.convert_time_unit(duration, :native, :microsecond)
///   {time, value}
/// end
/// ```
#[locate_code]
fn code(arc_process: &Arc<Process>) -> code::Result {
    arc_process.reduce();

    let module = arc_process.stack_peek(1).unwrap();
    let function = arc_process.stack_peek(2).unwrap();
    let arguments = arc_process.stack_peek(3).unwrap();

    arc_process.stack_popn(3);

    label_1::place_frame_with_arguments(
        arc_process,
        Placement::Replace,
        module,
        function,
        arguments,
    )
    .unwrap();
    monotonic_time_0::place_frame_with_arguments(arc_process, Placement::Push).unwrap();

    Process::call_code(arc_process)
}

fn frame(location: Location, code: Code) -> Frame {
    Frame::new(module(), function(), ARITY, location, code)
}

fn function() -> Atom {
    Atom::try_from_str("t3").unwrap()
}
