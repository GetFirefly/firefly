use std::sync::Arc;

use liblumen_alloc::erts::exception::Alloc;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::{code, Process};
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::ModuleFunctionArity;

use super::label_6;

pub fn place_frame_with_arguments(
    process: &Process,
    placement: Placement,
    document: Term,
    parent: Term,
) -> Result<(), Alloc> {
    process.stack_push(parent)?;
    process.stack_push(document)?;
    process.place_frame(frame(), placement);

    Ok(())
}

// Private

// ```elixir
// # label 5
// # pushed to stack: (document, parent)
// # returned form call: :ok
// # full stack: (:ok, document, parent)
// # returns: {:ok, new_child}
// {:ok, new_child} = Lumen.Web.Document.create_element(document, "ul");
// {:ok, inserted_child} = Lumen.Web.insert_before(parent, new_child, nil)
// ```
fn code(arc_process: &Arc<Process>) -> code::Result {
    arc_process.reduce();

    let ok = arc_process.stack_pop().unwrap();
    assert_eq!(ok, Atom::str_to_term("ok"));

    let document = arc_process.stack_pop().unwrap();
    assert!(document.is_resource_reference());

    let parent = arc_process.stack_pop().unwrap();
    assert!(parent.is_resource_reference());

    label_6::place_frame_with_arguments(arc_process, Placement::Replace, parent)?;

    let new_child_tag = arc_process.binary_from_str("ul")?;
    lumen_web::document::create_element_2::place_frame_with_arguments(
        arc_process,
        Placement::Push,
        document,
        new_child_tag,
    )?;

    Process::call_code(arc_process)
}

fn frame() -> Frame {
    let module_function_arity = Arc::new(ModuleFunctionArity {
        module: super::module(),
        function: super::function(),
        arity: 0,
    });

    Frame::new(module_function_arity, code)
}
