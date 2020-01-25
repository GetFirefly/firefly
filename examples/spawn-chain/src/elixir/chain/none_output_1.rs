use std::sync::Arc;

use liblumen_alloc::erts::exception::Alloc;
use liblumen_alloc::erts::process::{code, Process};
use liblumen_alloc::erts::term::closure::Definition;
use liblumen_alloc::erts::term::prelude::*;

use locate_code::locate_code;

pub fn closure(process: &Process) -> Result<Term, Alloc> {
    let definition = Definition::Export {
        function: function(),
    };
    process.closure_with_env_from_slice(super::module(), definition, ARITY, Some(LOCATED_CODE), &[])
}

// Private

const ARITY: u8 = 1;

/// defp none_output(_text) do
///   :ok
/// end
#[locate_code]
fn code(arc_process: &Arc<Process>) -> code::Result {
    arc_process.reduce();

    let _text = arc_process.stack_peek(1).unwrap();

    Process::return_from_call(arc_process, 1, Atom::str_to_term("ok"))?;

    Process::call_code(arc_process)
}

fn function() -> Atom {
    Atom::try_from_str("none_output").unwrap()
}
