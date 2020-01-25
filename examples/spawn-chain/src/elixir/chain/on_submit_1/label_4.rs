use std::convert::TryInto;
use std::sync::Arc;

use liblumen_alloc::atom;
use liblumen_alloc::erts::process::code::stack::frame::Placement;
use liblumen_alloc::erts::process::{code, Process};
use liblumen_alloc::erts::term::prelude::Encoded;

use locate_code::locate_code;

use lumen_runtime::otp::erlang;

use super::frame;

/// ```elixir
/// # label: 4
/// # pushed to stack: ()
/// # returned from call: n
/// # full stack: (n)
/// # returns: {time, value}
/// :erlang.spawn_opt(Chain, dom, [n], [min_heap_size: 79 + n * 10])
/// ```
pub fn place_frame(process: &Process, placement: Placement) {
    process.place_frame(frame(LOCATION, code), placement);
}

// Private

#[locate_code]
fn code(arc_process: &Arc<Process>) -> code::Result {
    arc_process.reduce();

    let n = arc_process.stack_peek(1).unwrap();
    assert!(n.is_integer());
    let n_usize: usize = n.try_into().unwrap();

    let arguments = arc_process.list_from_slice(&[n])?;
    let min_heap_size_value = arc_process.integer(79 + n_usize * 10)?;
    let min_heap_size_entry =
        arc_process.tuple_from_slice(&[atom!("min_heap_size"), min_heap_size_value])?;
    let options = arc_process.list_from_slice(&[min_heap_size_entry])?;

    arc_process.stack_popn(1);

    erlang::spawn_opt_4::place_frame_with_arguments(
        arc_process,
        Placement::Replace,
        atom!("Elixir.Chain"),
        atom!("dom"),
        arguments,
        options,
    )
    .unwrap();

    Process::call_code(arc_process)
}
