use std::convert::TryInto;
use std::sync::Arc;

use liblumen_alloc::erts::process::frames::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::{frames, Process};
use liblumen_alloc::erts::term::prelude::*;

use super::label_3;

/// ```elixir
/// # label: 2
/// # pushed to stack: ()
/// # returned from call: {:ok, n_input}
/// # full stack: ({:ok, n_input})
/// # returns: value_string
/// value_string = Lumen.Web.HTMLInputElement.value(n_input)
/// n = :erlang.binary_to_integer(value_string)
/// dom(n)
/// ```
pub fn place_frame(process: &Process, placement: Placement) {
    process.place_frame(frame(process), placement);
}

// Private

fn code(arc_process: &Arc<Process>) -> frames::Result {
    arc_process.reduce();

    let ok_n_input = arc_process.stack_pop().unwrap();
    let ok_n_input_tuple: Boxed<Tuple> = ok_n_input
        .try_into()
        .unwrap_or_else(|_| panic!("ok_n_input ({:?}) is not a tuple", ok_n_input));
    assert_eq!(ok_n_input_tuple.len(), 2);
    assert_eq!(ok_n_input_tuple[0], Atom::str_to_term("ok"));
    let n_input = ok_n_input_tuple[1];
    assert!(n_input.is_boxed_resource_reference());

    // ```elixir
    // # label: 3
    // # pushed to stack: ()
    // # returned from call: value_string
    // # full stack: (value_string)
    // # returns: n
    // n = :erlang.binary_to_integer(value_string)
    // dom(n)
    // ```
    label_3::place_frame(arc_process, Placement::Replace);

    liblumen_web::html_input_element::value_1::place_frame_with_arguments(
        arc_process,
        Placement::Push,
        n_input,
    )
    .unwrap();

    Process::call_native_or_yield(arc_process)
}

fn frame(process: &Process) -> Frame {
    let module_function_arity = process.current_module_function_arity().unwrap();

    Frame::new(module_function_arity, code)
}
