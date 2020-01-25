use std::sync::Arc;

use liblumen_alloc::erts::exception::Alloc;
use liblumen_alloc::erts::process::code::stack::frame::Placement;
use liblumen_alloc::erts::process::{code, Process};
use liblumen_alloc::erts::term::prelude::*;

use locate_code::locate_code;

use super::{frame, label_11};

pub fn place_frame_with_arguments(
    process: &Process,
    placement: Placement,
    document: Term,
    tr: Term,
    text_td: Term,
) -> Result<(), Alloc> {
    process.stack_push(text_td)?;
    process.stack_push(tr)?;
    process.stack_push(document)?;
    process.place_frame(frame(LOCATION, code), placement);

    Ok(())
}

// Private

/// ```elixir
/// # label 10
/// # pushed to stack: (document, tr, text_td)
/// # returned from call: :ok
/// # full stack: (:ok, document, tr, text_td)
/// # returns: :ok
/// Lumen::Web::Node.append_child(tr, text_td)
///
/// {:ok, tbody} = Lumen::Web::Document.get_element_by_id(document, "output")
/// Lumen::Web::Node.append_child(tbody, tr)
/// ```
#[locate_code]
fn code(arc_process: &Arc<Process>) -> code::Result {
    arc_process.reduce();

    let ok = arc_process.stack_pop().unwrap();
    assert_eq!(ok, Atom::str_to_term("ok"));
    let document = arc_process.stack_pop().unwrap();
    assert!(document.is_boxed_resource_reference());
    let tr = arc_process.stack_pop().unwrap();
    assert!(tr.is_boxed_resource_reference());
    let text_td = arc_process.stack_pop().unwrap();
    assert!(text_td.is_boxed_resource_reference());

    label_11::place_frame_with_arguments(arc_process, Placement::Replace, document, tr).unwrap();
    lumen_web::node::append_child_2::place_frame_with_arguments(
        arc_process,
        Placement::Push,
        tr,
        text_td,
    )
    .unwrap();

    Process::call_code(arc_process)
}
