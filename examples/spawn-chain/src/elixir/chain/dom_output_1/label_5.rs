use std::convert::TryInto;
use std::sync::Arc;

use liblumen_alloc::erts::exception::Alloc;
use liblumen_alloc::erts::process::frames::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::{frames, Process};
use liblumen_alloc::erts::term::prelude::*;

use crate::elixir::chain::dom_output_1::label_6;

pub fn place_frame_with_arguments(
    process: &Process,
    placement: Placement,
    document: Term,
    tr: Term,
    pid_text: Term,
    text: Term,
) -> Result<(), Alloc> {
    process.stack_push(text)?;
    process.stack_push(pid_text)?;
    process.stack_push(tr)?;
    process.stack_push(document)?;
    process.place_frame(frame(process), placement);

    Ok(())
}

// Private

/// ```elixir
/// # label 5
/// # pushed to stack: (document, tr, pid_text, text)
/// # returned from call: {:ok, pid_td}
/// # full stack: ({:ok, pid_td}, document, tr, pid_text, text)
/// # returns: :ok
/// Lumen::Web::Node.append_child(pid_td, pid_text)
/// Lumen::Web::Node.append_child(tr, pid_td)
///
/// {:ok, text_text} = Lumen::Web::Document.create_text_node(document, to_string(text))
/// {:ok, text_td} = Lumen::Web::Document.create_element(document, "td")
/// Lumen::Web::Node.append_child(text_td, text_text)
/// Lumen::Web::Node.append_child(tr, text_td)
///
/// {:ok, tbody} = Lumen::Web::Document.get_element_by_id(document, "output")
/// Lumen::Web::Node.append_child(tbody, tr)
/// ```
fn code(arc_process: &Arc<Process>) -> frames::Result {
    arc_process.reduce();

    let ok_pid_td = arc_process.stack_pop().unwrap();
    assert!(ok_pid_td.is_boxed_tuple());
    let document = arc_process.stack_pop().unwrap();
    assert!(document.is_boxed_resource_reference());
    let tr = arc_process.stack_pop().unwrap();
    assert!(tr.is_boxed_resource_reference());
    let pid_text = arc_process.stack_pop().unwrap();
    assert!(pid_text.is_boxed_resource_reference());
    let text = arc_process.stack_pop().unwrap();

    let ok_pid_td_tuple: Boxed<Tuple> = ok_pid_td.try_into().unwrap();
    assert_eq!(ok_pid_td_tuple.len(), 2);
    assert_eq!(ok_pid_td_tuple[0], Atom::str_to_term("ok"));
    let pid_td = ok_pid_td_tuple[1];
    assert!(pid_td.is_boxed_resource_reference());

    label_6::place_frame_with_arguments(
        arc_process,
        Placement::Replace,
        document,
        tr,
        pid_td,
        text,
    )
    .unwrap();
    liblumen_web::node::append_child_2::place_frame_with_arguments(
        arc_process,
        Placement::Push,
        pid_td,
        pid_text,
    )
    .unwrap();

    Process::call_native_or_yield(arc_process)
}

fn frame(process: &Process) -> Frame {
    let module_function_arity = process.current_module_function_arity().unwrap();

    Frame::new(module_function_arity, code)
}
