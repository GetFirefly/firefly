use std::convert::TryInto;
use std::sync::Arc;

use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::{code, ProcessControlBlock};
use liblumen_alloc::erts::term::{atom_unchecked, Boxed, Tuple};

use super::label_2;

/// ```elixir
/// # label: 1
/// # pushed to stack: ()
/// # returned from call: {:ok, event_target}
/// # full stack: ({:ok, event_target})
/// # returns: {:ok, n_input}
/// {:ok, n_input} = Lumen.Web.HTMLFormElement.element(event_target, "n")
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

    let ok_event_target = arc_process.stack_pop().unwrap();
    assert!(
        ok_event_target.is_tuple(),
        "ok_event_target ({:?}) is not a tuple",
        ok_event_target
    );
    let ok_event_target_tuple: Boxed<Tuple> = ok_event_target.try_into().unwrap();
    assert_eq!(ok_event_target_tuple.len(), 2);
    assert_eq!(ok_event_target_tuple[0], atom_unchecked("ok"));
    let event_target = ok_event_target_tuple[1];
    assert!(event_target.is_resource_reference());

    // ```elixir
    // # label: 2
    // # pushed to stack: ()
    // # returned from call: {:ok, n_input}
    // # full stack: ({:ok, n_input})
    // # returns: value_string
    // value_string = Lumen.Web.HTMLInputElement.value(n_input)
    // n = :erlang.binary_to_integer(value_string)
    // dom(n)
    // ```
    label_2::place_frame(arc_process, Placement::Replace);

    let name = arc_process.binary_from_str("n")?;
    lumen_web::html_form_element::element_2::place_frame_with_arguments(
        arc_process,
        Placement::Push,
        event_target,
        name,
    )?;

    ProcessControlBlock::call_code(arc_process)
}

fn frame(process: &ProcessControlBlock) -> Frame {
    let module_function_arity = process.current_module_function_arity().unwrap();

    Frame::new(module_function_arity, code)
}
