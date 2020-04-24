use std::sync::Arc;

use liblumen_alloc::erts::exception::Alloc;
use liblumen_alloc::erts::process::frames::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::{frames, Process};
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::erts::ModuleFunctionArity;

/// ```elixir
/// defp reduce_range_dec(first, first, acc, fun) do
///   fun.(first, acc)
/// end
///
/// defp reduce_range_dec(first, last, acc, fun) do
///   reduce_range_dec(first - 1, last, fun.(first, acc), fun)
/// end
/// ```
pub fn place_frame_with_arguments(
    process: &Process,
    placement: Placement,
    first: Term,
    last: Term,
    acc: Term,
    fun: Term,
) -> Result<(), Alloc> {
    process.stack_push(fun)?;
    process.stack_push(acc)?;
    process.stack_push(last)?;
    process.stack_push(first)?;
    process.place_frame(frame(), placement);

    Ok(())
}

fn code(_arc_process: &Arc<Process>) -> frames::Result {
    unimplemented!()
}

fn frame() -> Frame {
    Frame::new(module_function_arity(), code)
}

fn function() -> Atom {
    Atom::try_from_str("reduce_range_dec").unwrap()
}

fn module_function_arity() -> Arc<ModuleFunctionArity> {
    Arc::new(ModuleFunctionArity {
        module: super::module(),
        function: function(),
        arity: 3,
    })
}
