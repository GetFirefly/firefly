use std::sync::Arc;

use liblumen_alloc::erts::exception::Alloc;
use liblumen_alloc::erts::process::{frames, Process};
use liblumen_alloc::erts::term::prelude::*;

pub fn closure(process: &Process) -> Result<Term, Alloc> {
    process.export_closure(super::module(), function(), ARITY, Some(code))
}

// Private

const ARITY: u8 = 1;

/// defp none_output(_text) do
///   :ok
/// end
fn code(arc_process: &Arc<Process>) -> frames::Result {
    arc_process.reduce();

    let _text = arc_process.stack_peek(1).unwrap();

    Process::return_from_call(arc_process, 1, Atom::str_to_term("ok"))?;

    Process::call_native_or_yield(arc_process)
}

fn function() -> Atom {
    Atom::try_from_str("none_output").unwrap()
}
