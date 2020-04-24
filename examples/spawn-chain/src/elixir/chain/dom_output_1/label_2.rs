use std::convert::TryInto;
use std::sync::Arc;

use liblumen_alloc::erts::exception::Alloc;
use liblumen_alloc::erts::process::frames::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::{frames, Process};
use liblumen_alloc::erts::term::prelude::*;

use crate::elixir::chain::dom_output_1::label_3;

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
/// # label 2
/// # pushed to stack: (text)
/// # returned from call: {:ok, document}
/// # full stack: ({:ok, document}, text)
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

    let ok_document = arc_process.stack_pop().unwrap();
    assert!(
        ok_document.is_boxed_tuple(),
        "ok_document ({:?}) is not a tuple",
        ok_document
    );
    let text = arc_process.stack_pop().unwrap();

    let ok_document_tuple: Boxed<Tuple> = ok_document.try_into().unwrap();
    assert_eq!(ok_document_tuple.len(), 2);
    assert_eq!(ok_document_tuple[0], Atom::str_to_term("ok"));
    let document = ok_document_tuple[1];
    assert!(document.is_boxed_resource_reference());

    label_3::place_frame_with_arguments(arc_process, Placement::Replace, document, text).unwrap();

    let tag = arc_process.binary_from_str("tr").unwrap();
    liblumen_web::document::create_element_2::place_frame_with_arguments(
        arc_process,
        Placement::Push,
        document,
        tag,
    )
    .unwrap();

    Process::call_native_or_yield(arc_process)
}

fn frame(process: &Process) -> Frame {
    let module_function_arity = process.current_module_function_arity().unwrap();

    Frame::new(module_function_arity, code)
}
