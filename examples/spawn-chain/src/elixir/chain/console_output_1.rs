mod label_1;

use std::sync::Arc;

use liblumen_alloc::erts::exception::Alloc;
use liblumen_alloc::erts::process::frames::stack::frame::Placement;
use liblumen_alloc::erts::process::{frames, Process};
use liblumen_alloc::erts::term::prelude::*;

use liblumen_otp::erlang;

pub fn closure(process: &Process) -> Result<Term, Alloc> {
    process.export_closure(function(), super::module(), ARITY, Some(code))
}

const ARITY: u8 = 1;

/// defp console_output(text) do
///   IO.puts("#{self()} #{text}")
/// end
fn code(arc_process: &Arc<Process>) -> frames::Result {
    arc_process.reduce();

    let text = arc_process.stack_pop().unwrap();

    label_1::place_frame_with_arguments(arc_process, Placement::Replace, text).unwrap();
    erlang::self_0::place_frame_with_arguments(arc_process, Placement::Push).unwrap();

    Process::call_native_or_yield(arc_process)
}

fn function() -> Atom {
    Atom::try_from_str("console_output").unwrap()
}
