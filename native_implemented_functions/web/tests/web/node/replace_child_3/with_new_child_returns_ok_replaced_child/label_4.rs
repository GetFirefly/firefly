use std::sync::Arc;

use liblumen_alloc::erts::exception::Alloc;
use liblumen_alloc::erts::process::frames::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::{frames, Process};
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::ModuleFunctionArity;

use super::label_5;

pub fn place_frame_with_arguments(
    process: &Process,
    placement: Placement,
    document: Term,
    parent: Term,
    old_child: Term,
) -> Result<(), Alloc> {
    process.stack_push(old_child)?;
    process.stack_push(parent)?;
    process.stack_push(document)?;
    process.place_frame(frame(), placement);

    Ok(())
}

// Private

// ```elixir
// # label 4
// # pushed to stack: (document, parent, old_child)
// # returned form call: :ok
// # full stack: (:ok, document, parent, old_child)
// # returns: {:ok, new_child}
// {:ok, new_child} = Lumen.Web.Document.create_element(document, "ul");
// {:ok, replaced_child} = Lumen.Web.replace_child(parent, new_child, old_child)
// ```
fn code(arc_process: &Arc<Process>) -> frames::Result {
    arc_process.reduce();

    let ok = arc_process.stack_pop().unwrap();
    assert_eq!(ok, Atom::str_to_term("ok"));

    let document = arc_process.stack_pop().unwrap();
    assert!(document.is_boxed_resource_reference());

    let parent = arc_process.stack_pop().unwrap();
    assert!(parent.is_boxed_resource_reference());

    let old_child = arc_process.stack_pop().unwrap();
    assert!(old_child.is_boxed_resource_reference());

    label_5::place_frame_with_arguments(arc_process, Placement::Replace, parent, old_child)?;

    let new_child_tag = arc_process.binary_from_str("ul")?;
    liblumen_web::document::create_element_2::place_frame_with_arguments(
        arc_process,
        Placement::Push,
        document,
        new_child_tag,
    )?;

    Process::call_native_or_yield(arc_process)
}

fn frame() -> Frame {
    let module_function_arity = Arc::new(ModuleFunctionArity {
        module: super::module(),
        function: super::function(),
        arity: 0,
    });

    Frame::new(module_function_arity, code)
}
