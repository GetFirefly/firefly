use std::convert::TryInto;
use std::sync::Arc;

use liblumen_alloc::erts::exception::Alloc;
use liblumen_alloc::erts::process::frames::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::{frames, Process};
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
// # pushed to stack: (parent_document)
// # returned form call: {:ok, old_child}
// # full stack: ({:ok, old_child}, parent_document)
// # returns: {:ok, parent}
// {:ok, parent} = Lumen.Web.Document.create_element(parent_document, "div")
// :ok = Lumen.Web.Node.append_child(parent, old_child)
// {:ok, new_child} = Lumen.Web.Document.create_element(document, "ul");
// {:ok, replaced_child} = Lumen.Web.replace_child(parent, new_child, old_child)
// ```
fn code(arc_process: &Arc<Process>) -> frames::Result {
    arc_process.reduce();

    let ok_old_child = arc_process.stack_pop().unwrap();
    assert!(
        ok_old_child.is_boxed_tuple(),
        "ok_old_child ({:?}) is not a tuple",
        ok_old_child
    );
    let ok_old_child_tuple: Boxed<Tuple> = ok_old_child.try_into().unwrap();
    assert_eq!(ok_old_child_tuple.len(), 2);
    assert_eq!(ok_old_child_tuple[0], Atom::str_to_term("ok"));
    let old_child = ok_old_child_tuple[1];
    assert!(old_child.is_boxed_resource_reference());

    let document = arc_process.stack_pop().unwrap();
    assert!(document.is_boxed_resource_reference());

    label_3::place_frame_with_arguments(arc_process, Placement::Replace, document, old_child)?;

    let parent_tag = arc_process.binary_from_str("div")?;
    liblumen_web::document::create_element_2::place_frame_with_arguments(
        arc_process,
        Placement::Push,
        document,
        parent_tag,
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
