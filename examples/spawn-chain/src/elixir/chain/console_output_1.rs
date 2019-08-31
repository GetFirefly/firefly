mod label_1;

use std::sync::Arc;

use liblumen_alloc::erts::exception::system::Alloc;
use liblumen_alloc::erts::process::code::stack::frame::Placement;
use liblumen_alloc::erts::process::{code, Process};
use liblumen_alloc::erts::term::{Atom, Term};
use liblumen_alloc::erts::ModuleFunctionArity;

use lumen_runtime::otp::erlang;

pub fn closure(process: &Process) -> Result<Term, Alloc> {
    process.closure(process.pid_term(), module_function_arity(), code, vec![])
}

/// defp console_output(text) do
///   IO.puts("#{self()} #{text}")
/// end
fn code(arc_process: &Arc<Process>) -> code::Result {
    arc_process.reduce();

    let text = arc_process.stack_pop().unwrap();

    label_1::place_frame_with_arguments(arc_process, Placement::Replace, text)?;
    erlang::self_0::place_frame(arc_process, Placement::Push);

    Process::call_code(arc_process)
}

fn function() -> Atom {
    Atom::try_from_str("console_output").unwrap()
}

fn module_function_arity() -> Arc<ModuleFunctionArity> {
    Arc::new(ModuleFunctionArity {
        module: super::module(),
        function: function(),
        arity: 1,
    })
}
