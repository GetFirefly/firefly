use std::sync::Arc;

use liblumen_alloc::erts::exception::system::Alloc;
use liblumen_alloc::erts::process::{code, Process};
use liblumen_alloc::erts::term::{atom_unchecked, Atom, Term};

pub fn closure(process: &Process) -> Result<Term, Alloc> {
    process.export_closure(super::module(), function(), ARITY, Some(code))
}

// Private

const ARITY: u8 = 1;

/// defp none_output(_text) do
///   :ok
/// end
fn code(arc_process: &Arc<Process>) -> code::Result {
    arc_process.reduce();

    let _text = arc_process.stack_pop().unwrap();

    Process::return_from_call(arc_process, atom_unchecked("ok")).unwrap();

    Process::call_code(arc_process)
}

fn function() -> Atom {
    Atom::try_from_str("none_output").unwrap()
}
