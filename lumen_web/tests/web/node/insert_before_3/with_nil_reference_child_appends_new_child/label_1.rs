use std::convert::TryInto;
use std::sync::Arc;

use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::{code, Process};
use liblumen_alloc::erts::term::{atom_unchecked, Boxed, Tuple};
use liblumen_alloc::ModuleFunctionArity;

use super::label_2;

pub fn place_frame(process: &Process, placement: Placement) {
    process.place_frame(frame(), placement);
}

// Private

// ```elixir
// # label 1
// # pushed to stack: ()
// # returned form call: {:ok, document}
// # full stack: ({:ok, document})
// # returns: {:ok, existing_child}
// {:ok, existing_child} = Lumen.Web.Document.create_element(document, "table")
// {:ok, parent} = Lumen.Web.Document.create_element(parent_document, "div")
// :ok = Lumen.Web.Node.append_child(document, parent)
// :ok = Lumen.Web.Node.append_child(parent, existing_child)
// {:ok, new_child} = Lumen.Web.Document.create_element(document, "ul");
// {:ok, inserted_child} = Lumen.Web.insert_before(parent, new_child, nil)
// ```
fn code(arc_process: &Arc<Process>) -> code::Result {
    arc_process.reduce();

    let ok_document = arc_process.stack_pop().unwrap();
    assert!(
        ok_document.is_tuple(),
        "ok_document ({:?}) is not a tuple",
        ok_document
    );
    let ok_document_tuple: Boxed<Tuple> = ok_document.try_into().unwrap();
    assert_eq!(ok_document_tuple.len(), 2);
    assert_eq!(ok_document_tuple[0], atom_unchecked("ok"));
    let document = ok_document_tuple[1];
    assert!(document.is_resource_reference());

    label_2::place_frame_with_arguments(arc_process, Placement::Replace, document)?;

    let existing_child_tag = arc_process.binary_from_str("table")?;
    lumen_web::document::create_element_2::place_frame_with_arguments(
        arc_process,
        Placement::Push,
        document,
        existing_child_tag,
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
