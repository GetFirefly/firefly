use std::convert::TryInto;
use std::sync::Arc;

use liblumen_alloc::erts::exception::Alloc;
use liblumen_alloc::erts::process::frames::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::{frames, Process};
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::ModuleFunctionArity;

use web_sys::Element;

use super::label_5;

pub fn place_frame_with_arguments(
    process: &Process,
    placement: Placement,
    document: Term,
    parent: Term,
    existing_child: Term,
) -> Result<(), Alloc> {
    process.stack_push(existing_child)?;
    process.stack_push(parent)?;
    process.stack_push(document)?;
    process.place_frame(frame(), placement);

    Ok(())
}

// Private

// ```elixir
// # label 4
// # pushed to stack: (document, parent, existing_child)
// # returned form call: :ok
// # full stack: (:ok, document, parent, existing_child)
// # returns: :ok
// :ok = Lumen.Web.Node.append_child(parent, existing_child)
// {:ok, new_child} = Lumen.Web.Document.create_element(document, "ul");
// {:ok, inserted_child} = Lumen.Web.insert_before(parent, new_child, nil)
// ```
fn code(arc_process: &Arc<Process>) -> frames::Result {
    arc_process.reduce();

    let ok = arc_process.stack_pop().unwrap();
    assert_eq!(ok, Atom::str_to_term("ok"));

    let document = arc_process.stack_pop().unwrap();
    assert!(document.is_boxed_resource_reference());

    let parent = arc_process.stack_pop().unwrap();
    assert!(parent.is_boxed_resource_reference());

    let existing_child = arc_process.stack_pop().unwrap();
    let existing_child_ref: Boxed<Resource> = existing_child.try_into().unwrap();
    let existing_child_reference: Resource = existing_child_ref.into();
    let _: &Element = existing_child_reference.downcast_ref().unwrap();

    label_5::place_frame_with_arguments(arc_process, Placement::Replace, document, parent)?;

    liblumen_web::node::append_child_2::place_frame_with_arguments(
        arc_process,
        Placement::Push,
        parent,
        existing_child,
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
