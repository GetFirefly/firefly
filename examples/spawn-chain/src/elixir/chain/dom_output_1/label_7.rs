use std::sync::Arc;

use liblumen_alloc::erts::exception::Alloc;
use liblumen_alloc::erts::process::frames::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::{frames, Process};
use liblumen_alloc::erts::term::prelude::*;

use crate::elixir::chain::dom_output_1::label_8;

pub fn place_frame_with_arguments(
    process: &Process,
    placement: Placement,
    document: Term,
    tr: Term,
    text: Term,
) -> Result<(), Alloc> {
    process.stack_push(text)?;
    process.stack_push(tr)?;
    process.stack_push(document)?;
    process.place_frame(frame(process), placement);

    Ok(())
}

// Private

/// ```elixir
/// # label 7
/// # pushed to stack: (document, tr, text)
/// # returned from call: :ok
/// # full stack: (:ok, document, tr, text)
/// # returns: text_text
/// {:ok, text_text} = Lumen::Web::Document.create_text_node(document, text)
/// {:ok, text_td} = Lumen::Web::Document.create_element(document, "td")
/// Lumen::Web::Node.append_child(text_td, text_text)
/// Lumen::Web::Node.append_child(tr, text_td)
///
/// {:ok, tbody} = Lumen::Web::Document.get_element_by_id(document, "output")
/// Lumen::Web::Node.append_child(tbody, tr)
/// ```
fn code(arc_process: &Arc<Process>) -> frames::Result {
    arc_process.reduce();

    let ok = arc_process.stack_pop().unwrap();
    assert_eq!(ok, Atom::str_to_term("ok"));
    let document = arc_process.stack_pop().unwrap();
    assert!(document.is_boxed_resource_reference());
    let tr = arc_process.stack_pop().unwrap();
    assert!(tr.is_boxed_resource_reference());
    let text = arc_process.stack_pop().unwrap();

    label_8::place_frame_with_arguments(arc_process, Placement::Replace, document, tr).unwrap();
    liblumen_web::document::create_text_node_2::place_frame_with_arguments(
        arc_process,
        Placement::Push,
        document,
        text,
    )
    .unwrap();

    Process::call_native_or_yield(arc_process)
}

fn frame(process: &Process) -> Frame {
    let module_function_arity = process.current_module_function_arity().unwrap();

    Frame::new(module_function_arity, code)
}
