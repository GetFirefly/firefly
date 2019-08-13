use std::sync::Arc;

use liblumen_alloc::erts::exception::system::Alloc;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::{code, ProcessControlBlock};
use liblumen_alloc::erts::term::{atom_unchecked, Term};

use crate::elixir::chain::dom_output_1::label_12;

pub fn place_frame_with_arguments(
    process: &ProcessControlBlock,
    placement: Placement,
    document: Term,
    tr: Term,
) -> Result<(), Alloc> {
    process.stack_push(tr)?;
    process.stack_push(document)?;
    process.place_frame(frame(process), placement);

    Ok(())
}

// Private

/// ```elixir
/// # label 10
/// # pushed to stack: (document, tr)
/// # returned from call: :ok
/// # full stack: (:ok, document, tr)
/// # returns: {:ok, tbody}
/// {:ok, tbody} = Lumen::Web::Document.get_element_by_id(document, "output")
/// Lumen::Web::Node.append_child(tbody, tr)
/// ```
fn code(arc_process: &Arc<ProcessControlBlock>) -> code::Result {
    arc_process.reduce();

    let ok = arc_process.stack_pop().unwrap();
    assert_eq!(ok, atom_unchecked("ok"));
    let document = arc_process.stack_pop().unwrap();
    assert!(document.is_resource_reference());
    let tr = arc_process.stack_pop().unwrap();
    assert!(tr.is_resource_reference());

    let id = arc_process.binary_from_str("output")?;
    lumen_web::document::get_element_by_id_2::place_frame_with_arguments(
        arc_process,
        Placement::Replace,
        document,
        id,
    )?;

    label_12::place_frame_with_arguments(arc_process, Placement::Push, document, tr)?;

    ProcessControlBlock::call_code(arc_process)
}

fn frame(process: &ProcessControlBlock) -> Frame {
    let module_function_arity = process.current_module_function_arity().unwrap();

    Frame::new(module_function_arity, code)
}
