use std::convert::TryInto;
use std::sync::Arc;

use liblumen_alloc::erts::exception::system::Alloc;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::{code, Process};
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::ModuleFunctionArity;

use super::label_3;

pub fn place_frame_with_arguments(
    process: &Process,
    placement: Placement,
    document: Term,
) -> Result<(), Alloc> {
    process.stack_push(document)?;
    process.place_frame(frame(), placement);

    Ok(())
}

// Private

// ```elixir
// # label 2
// # pushed to stack: (document)
// # returned form call: {:ok, existing_child}
// # full stack: ({:ok, existing_child}, document)
// # returns: {:ok, parent}
// {:ok, parent} = Lumen.Web.Document.create_element(parent_document, "div")
// :ok = Lumen.Web.Node.append_child(document, parent)
// :ok = Lumen.Web.Node.append_child(parent, existing_child)
// {:ok, new_child} = Lumen.Web.Document.create_element(document, "ul");
// {:ok, inserted_child} = Lumen.Web.insert_before(parent, new_child, nil)
// ```
fn code(arc_process: &Arc<Process>) -> code::Result {
    arc_process.reduce();

    let ok_existing_child = arc_process.stack_pop().unwrap();
    assert!(
        ok_existing_child.is_tuple(),
        "ok_existing_child ({:?}) is not a tuple",
        ok_existing_child
    );
    let ok_existing_child_tuple: Boxed<Tuple> = ok_existing_child.try_into().unwrap();
    assert_eq!(ok_existing_child_tuple.len(), 2);
    assert_eq!(ok_existing_child_tuple[0], Atom::str_to_term("ok"));
    let existing_child = ok_existing_child_tuple[1];
    assert!(existing_child.is_resource_reference());

    let document = arc_process.stack_pop().unwrap();
    assert!(document.is_resource_reference());

    label_3::place_frame_with_arguments(arc_process, Placement::Replace, document, existing_child)?;

    let parent_tag = arc_process.binary_from_str("div")?;
    lumen_web::document::create_element_2::place_frame_with_arguments(
        arc_process,
        Placement::Push,
        document,
        parent_tag,
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
