use std::convert::TryInto;
use std::sync::Arc;

use liblumen_alloc::erts::exception::system::Alloc;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::{code, ProcessControlBlock};
use liblumen_alloc::erts::term::{atom_unchecked, Boxed, Term, Tuple};

use crate::elixir::chain::dom_output_1::label_5;

pub fn place_frame_with_arguments(
    process: &ProcessControlBlock,
    placement: Placement,
    document: Term,
    tr: Term,
    text: Term,
) -> Result<(), Alloc> {
    assert!(text.is_binary());
    process.stack_push(text)?;
    process.stack_push(tr)?;
    process.stack_push(document)?;
    process.place_frame(frame(process), placement);

    Ok(())
}

// Private

/// ```elixir
/// # label 4
/// # pushed to stack: (document, tr, text)
/// # returned from call: {:ok, pid_text}
/// # full stack: ({:ok, pid_text}, document, tr, text)
/// # returns: {:ok, pid_td}
/// {:ok, pid_td} = Lumen::Web::Document.create_element(document, "td")
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
fn code(arc_process: &Arc<ProcessControlBlock>) -> code::Result {
    arc_process.reduce();

    let ok_pid_text = arc_process.stack_pop().unwrap();
    assert!(ok_pid_text.is_tuple());
    let document = arc_process.stack_pop().unwrap();
    assert!(document.is_resource_reference());
    let tr = arc_process.stack_pop().unwrap();
    assert!(tr.is_resource_reference());
    let text = arc_process.stack_pop().unwrap();

    let ok_pid_text_tuple: Boxed<Tuple> = ok_pid_text.try_into().unwrap();
    assert_eq!(ok_pid_text_tuple.len(), 2);
    assert_eq!(ok_pid_text_tuple[0], atom_unchecked("ok"));
    let pid_text = ok_pid_text_tuple[1];
    assert!(pid_text.is_resource_reference());

    let tag = arc_process.binary_from_str("td")?;
    lumen_web::document::create_element_2::place_frame_with_arguments(
        arc_process,
        Placement::Replace,
        document,
        tag,
    )?;

    label_5::place_frame_with_arguments(
        arc_process,
        Placement::Push,
        document,
        tr,
        pid_text,
        text,
    )?;

    ProcessControlBlock::call_code(arc_process)
}

fn frame(process: &ProcessControlBlock) -> Frame {
    let module_function_arity = process.current_module_function_arity().unwrap();

    Frame::new(module_function_arity, code)
}
