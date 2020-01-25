mod label_1;

use std::sync::Arc;

use liblumen_alloc::erts::exception::Alloc;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::code::{self, Code};
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::closure::Definition;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::location::Location;

use locate_code::locate_code;

use lumen_runtime::otp::erlang;

pub fn closure(process: &Process) -> Result<Term, Alloc> {
    let definition = Definition::Export {
        function: function(),
    };
    process.closure_with_env_from_slice(super::module(), definition, ARITY, Some(LOCATED_CODE), &[])
}

// Private

const ARITY: u8 = 1;

/// defp console_output(text) do
///   IO.puts("#{self()} #{text}")
/// end
#[locate_code]
fn code(arc_process: &Arc<Process>) -> code::Result {
    arc_process.reduce();

    let text = arc_process.stack_pop().unwrap();

    label_1::place_frame_with_arguments(arc_process, Placement::Replace, text).unwrap();
    erlang::self_0::place_frame_with_arguments(arc_process, Placement::Push).unwrap();

    Process::call_code(arc_process)
}

fn frame(location: Location, code: Code) -> Frame {
    Frame::new(super::module(), function(), ARITY, location, code)
}

fn function() -> Atom {
    Atom::try_from_str("console_output").unwrap()
}
