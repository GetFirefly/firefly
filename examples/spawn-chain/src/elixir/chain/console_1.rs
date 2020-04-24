use std::sync::Arc;

use liblumen_alloc::erts::process::frames::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::{frames, Process};
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::erts::{Arity, ModuleFunctionArity};

use crate::elixir::chain::{console_output_1, run_2};

pub fn export() {
    lumen_rt_full::code::export::insert(super::module(), function(), ARITY, code);
}

/// ```elixir
/// # pushed to stack: (n)
/// # returned from call: N/A
/// # full stack: (n)
/// # returns: final_answer
/// def console(n) do
///   run(n, &console_output/1)
/// end
/// ```
pub fn place_frame_with_arguments(
    process: &Process,
    placement: Placement,
    n: Term,
) -> frames::Result {
    assert!(n.is_integer());
    process.stack_push(n)?;
    process.place_frame(frame(), placement);

    Ok(())
}

// Private

const ARITY: Arity = 1;

fn code(arc_process: &Arc<Process>) -> frames::Result {
    arc_process.reduce();

    let n = arc_process.stack_pop().unwrap();
    assert!(n.is_integer());

    let console_output_closure = console_output_1::closure(arc_process).unwrap();
    run_2::place_frame_with_arguments(arc_process, Placement::Replace, n, console_output_closure)
        .unwrap();

    Process::call_native_or_yield(arc_process)
}

fn frame() -> Frame {
    Frame::new(module_function_arity(), code)
}

fn function() -> Atom {
    Atom::try_from_str("console").unwrap()
}

fn module_function_arity() -> Arc<ModuleFunctionArity> {
    Arc::new(ModuleFunctionArity {
        module: super::module(),
        function: function(),
        arity: ARITY,
    })
}
