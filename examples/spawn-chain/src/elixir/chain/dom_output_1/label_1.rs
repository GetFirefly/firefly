use std::convert::TryInto;
use std::sync::Arc;

use liblumen_alloc::erts::exception::Alloc;
use liblumen_alloc::erts::process::frames::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::{frames, Process};
use liblumen_alloc::erts::term::prelude::*;

use crate::elixir::chain::dom_output_1::label_2;

pub fn place_frame_with_arguments(
    process: &Process,
    placement: Placement,
    text: Term,
) -> Result<(), Alloc> {
    process.stack_push(text)?;
    process.place_frame(frame(process), placement);

    Ok(())
}

// Private

/// ```elixir
/// # label 1
/// # pushed to stack: (text)
/// # returned from call: {:ok, window}
/// # full stack: ({:ok, window}, text)
/// {:ok, document} = Lumen::Web::Window.document(window)
/// {:ok, tr} = Lumen::Web::Document.create_element(document, "tr")
///
/// {:ok, pid_text} = Lumen::Web::Document.create_text_node(document, to_string(self()))
/// {:ok, pid_td} = Lumen::Web::Document.create_element(document, "td")
/// Lumen::Web::Node.append_child(pid_td, pid_text);
/// Lumen::Web::Node.append_child(tr, pid_td)
///
/// {:ok, text_text} = Lumen::Web::Document.create_text_node(document, to_string(text()))
/// {:ok, text_td} = Lumen::Web::Document.create_element(document, "td")
/// Lumen::Web::Node.append_child(text_td, text_text);
/// Lumen::Web::Node.append_child(tr, text_td)
///
/// {:ok, tbody} = Lumen::Web::Document.get_element_by_id(document, "output")
/// Lumen::Web::Node.append_child(tbody, tr)
/// ```
fn code(arc_process: &Arc<Process>) -> frames::Result {
    arc_process.reduce();

    let ok_window = arc_process.stack_pop().unwrap();
    assert!(
        ok_window.is_boxed_tuple(),
        "ok_window ({:?}) is not a tuple",
        ok_window
    );
    let ok_window_tuple: Boxed<Tuple> = ok_window.try_into().unwrap();
    assert_eq!(ok_window_tuple.len(), 2);
    assert_eq!(ok_window_tuple[0], Atom::str_to_term("ok"));
    let window = ok_window_tuple[1];
    assert!(window.is_boxed_resource_reference());

    let text = arc_process.stack_pop().unwrap();
    assert!(text.is_binary());

    label_2::place_frame_with_arguments(arc_process, Placement::Replace, text).unwrap();
    liblumen_web::window::document_1::place_frame_with_arguments(
        arc_process,
        Placement::Push,
        window,
    )
    .unwrap();

    Process::call_native_or_yield(arc_process)
}

fn frame(process: &Process) -> Frame {
    let module_function_arity = process.current_module_function_arity().unwrap();

    Frame::new(module_function_arity, code)
}
