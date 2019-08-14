mod label_1;
mod label_2;
mod label_3;
mod label_4;
mod label_5;

use std::sync::Arc;

use liblumen_alloc::erts::exception::system::Alloc;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::{code, ProcessControlBlock};
use liblumen_alloc::erts::term::{Atom, Term};
use liblumen_alloc::ModuleFunctionArity;

use crate::otp::erlang::monotonic_time_0;

pub fn place_frame_with_arguments(
    process: &ProcessControlBlock,
    placement: Placement,
    module: Term,
    function: Term,
    arguments: Term,
) -> Result<(), Alloc> {
    process.stack_push(arguments)?;
    process.stack_push(function)?;
    process.stack_push(module)?;
    process.place_frame(frame(), placement);

    Ok(())
}

// Private

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
fn code(arc_process: &Arc<ProcessControlBlock>) -> code::Result {
    arc_process.reduce();

    let module = arc_process.stack_pop().unwrap();
    let function = arc_process.stack_pop().unwrap();
    let arguments = arc_process.stack_pop().unwrap();

    label_1::place_frame_with_arguments(
        arc_process,
        Placement::Replace,
        module,
        function,
        arguments,
    )?;
    monotonic_time_0::place_frame(arc_process, Placement::Push);

    ProcessControlBlock::call_code(arc_process)
}

fn frame() -> Frame {
    Frame::new(module_function_arity(), code)
}

fn function() -> Atom {
    Atom::try_from_str("t3").unwrap()
}

fn module_function_arity() -> Arc<ModuleFunctionArity> {
    Arc::new(ModuleFunctionArity {
        module: super::module(),
        function: function(),
        arity: 3,
    })
}
