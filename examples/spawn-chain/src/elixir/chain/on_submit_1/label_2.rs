use std::convert::TryInto;
use std::sync::Arc;

use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::{code, ProcessControlBlock};
use liblumen_alloc::erts::term::{atom_unchecked, Boxed, Tuple};

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
pub fn place_frame(process: &ProcessControlBlock, placement: Placement) {
    process.place_frame(frame(process), placement);
}

// Private

fn code(arc_process: &Arc<ProcessControlBlock>) -> code::Result {
    arc_process.reduce();

    let ok_n_input = arc_process.stack_pop().unwrap();
    assert!(
        ok_n_input.is_tuple(),
        "ok_n_input ({:?}) is not a tuple",
        ok_n_input
    );
    let ok_n_input_tuple: Boxed<Tuple> = ok_n_input.try_into().unwrap();
    assert_eq!(ok_n_input_tuple.len(), 2);
    assert_eq!(ok_n_input_tuple[0], atom_unchecked("ok"));
    let n_input = ok_n_input_tuple[1];
    assert!(n_input.is_resource_reference());

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

    lumen_web::html_input_element::value_1::place_frame_with_arguments(
        arc_process,
        Placement::Push,
        n_input,
    )?;

    ProcessControlBlock::call_code(arc_process)
}

fn frame(process: &ProcessControlBlock) -> Frame {
    let module_function_arity = process.current_module_function_arity().unwrap();

    Frame::new(module_function_arity, code)
}
