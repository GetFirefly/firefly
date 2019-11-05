use std::convert::TryInto;
use std::sync::Arc;

use liblumen_alloc::erts::exception::system::Alloc;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::{code, Process};
use liblumen_alloc::erts::term::{atom_unchecked, Boxed, Term, Tuple};

use crate::elixir::chain::dom_output_1::label_10;

pub fn place_frame_with_arguments(
    process: &Process,
    placement: Placement,
    document: Term,
    tr: Term,
    text_text: Term,
) -> Result<(), Alloc> {
    process.stack_push(text_text)?;
    process.stack_push(tr)?;
    process.stack_push(document)?;
    process.place_frame(frame(process), placement);

    Ok(())
}

// Private

/// ```elixir
/// # label 9
/// # pushed to stack: (document, tr, text_text)
/// # returned from call: {:ok, text_td}
/// # full stack: ({:ok, text_td}, document, tr, text_text)
/// # returns: :ok
/// Lumen::Web::Node.append_child(text_td, text_text)
/// Lumen::Web::Node.append_child(tr, text_td)
///
/// {:ok, tbody} = Lumen::Web::Document.get_element_by_id(document, "output")
/// Lumen::Web::Node.append_child(tbody, tr)
/// ```
fn code(arc_process: &Arc<Process>) -> code::Result {
    arc_process.reduce();

    let ok_text_td = arc_process.stack_pop().unwrap();
    assert!(ok_text_td.is_tuple());
    let document = arc_process.stack_pop().unwrap();
    assert!(document.is_resource_reference());
    let tr = arc_process.stack_pop().unwrap();
    assert!(tr.is_resource_reference());
    let text_text = arc_process.stack_pop().unwrap();
    assert!(text_text.is_resource_reference());

    let ok_text_td_tuple: Boxed<Tuple> = ok_text_td.try_into().unwrap();
    assert_eq!(ok_text_td_tuple.len(), 2);
    assert_eq!(ok_text_td_tuple[0], atom_unchecked("ok"));
    let text_td = ok_text_td_tuple[1];
    assert!(text_td.is_resource_reference());

    label_10::place_frame_with_arguments(arc_process, Placement::Replace, document, tr, text_td)
        .unwrap();
    lumen_web::node::append_child_2::place_frame_with_arguments(
        arc_process,
        Placement::Push,
        text_td,
        text_text,
    )
    .unwrap();

    Process::call_code(arc_process)
}

fn frame(process: &Process) -> Frame {
    let module_function_arity = process.current_module_function_arity().unwrap();

    Frame::new(module_function_arity, code)
}
