use std::convert::TryInto;
use std::sync::Arc;

use liblumen_alloc::erts::exception::Alloc;
use liblumen_alloc::erts::process::frames::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::{frames, Process};
use liblumen_alloc::erts::term::prelude::*;

use crate::elixir::chain::dom_output_1::label_4;

pub fn place_frame_with_arguments(
    process: &Process,
    placement: Placement,
    document: Term,
    text: Term,
) -> Result<(), Alloc> {
    process.stack_push(text)?;
    process.stack_push(document)?;
    process.place_frame(frame(process), placement);

    Ok(())
}

// Private

/// ```elixir
/// # label 3
/// # pushed to stack: (document, text)
/// # returned from call: {:ok, tr}
/// # full stack: ({:ok, tr}, document, text)
/// # returns: {:ok, pid_text}
/// {:ok, pid_text} = Lumen::Web::Document.create_text_node(document, to_string(self()))
/// {:ok, pid_td} = Lumen::Web::Document.create_element(document, "td")
/// Lumen::Web::Node.append_child(pid_td, pid_text)
/// Lumen::Web::Node.append_child(tr, pid_td)
///
/// {:ok, text_text} = Lumen::Web::Document.create_text_node(document, to_string(text()))
/// {:ok, text_td} = Lumen::Web::Document.create_element(document, "td")
/// Lumen::Web::Node.append_child(text_td, text_text)
/// Lumen::Web::Node.append_child(tr, text_td)
///
/// {:ok, tbody} = Lumen::Web::Document.get_element_by_id(document, "output")
/// Lumen::Web::Node.append_child(tbody, tr)
/// ```
fn code(arc_process: &Arc<Process>) -> frames::Result {
    arc_process.reduce();

    let ok_tr = arc_process.stack_pop().unwrap();
    assert!(ok_tr.is_boxed_tuple());
    let document = arc_process.stack_pop().unwrap();
    assert!(document.is_boxed_resource_reference());
    let text = arc_process.stack_pop().unwrap();

    let ok_tr_tuple: Boxed<Tuple> = ok_tr.try_into().unwrap();
    assert_eq!(ok_tr_tuple.len(), 2);
    assert_eq!(ok_tr_tuple[0], Atom::str_to_term("ok"));
    let tr = ok_tr_tuple[1];
    assert!(tr.is_boxed_resource_reference());

    label_4::place_frame_with_arguments(arc_process, Placement::Replace, document, tr, text)
        .unwrap();

    // TODO actually call `to_string(self)`
    let pid_text_binary = arc_process
        .binary_from_str(&format!("{}", arc_process.pid_term()))
        .unwrap();
    liblumen_web::document::create_text_node_2::place_frame_with_arguments(
        arc_process,
        Placement::Push,
        document,
        pid_text_binary,
    )
    .unwrap();

    Process::call_native_or_yield(arc_process)
}

fn frame(process: &Process) -> Frame {
    let module_function_arity = process.current_module_function_arity().unwrap();

    Frame::new(module_function_arity, code)
}
