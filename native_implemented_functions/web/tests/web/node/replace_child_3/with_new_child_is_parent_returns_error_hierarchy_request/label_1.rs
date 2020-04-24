use std::convert::TryInto;
use std::sync::Arc;

use liblumen_alloc::erts::process::frames::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::{frames, Process};
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::ModuleFunctionArity;

use super::label_2;

pub fn place_frame(process: &Process, placement: Placement) {
    process.place_frame(frame(), placement);
}

// Private

// ```elixir
// # label 1
// # pushed to stack: ("div")
// # returned form call: {:ok, document}
// # full stack: ({:ok, document}, "div")
// # returns: {:ok, old_child}
// {:ok, old_child} = Lumen.Web.Document.create_element(document, "table")
// {:ok, parent} = Lumen.Web.Document.create_element(document, "div")
// :ok = Lumen.Web.Node.append_child(parent, old_child)
// {:error, :hierarchy_request} = Lumen.Web.replace_child(parent, old_child, parent)
// ```
fn code(arc_process: &Arc<Process>) -> frames::Result {
    arc_process.reduce();

    let ok_document = arc_process.stack_pop().unwrap();
    assert!(
        ok_document.is_boxed_tuple(),
        "ok_document ({:?}) is not a tuple",
        ok_document
    );
    let ok_document_tuple: Boxed<Tuple> = ok_document.try_into().unwrap();
    assert_eq!(ok_document_tuple.len(), 2);
    assert_eq!(ok_document_tuple[0], Atom::str_to_term("ok"));
    let document = ok_document_tuple[1];
    assert!(document.is_boxed_resource_reference());

    label_2::place_frame_with_arguments(arc_process, Placement::Replace, document)?;

    let old_child_tag = arc_process.binary_from_str("table")?;
    liblumen_web::document::create_element_2::place_frame_with_arguments(
        arc_process,
        Placement::Push,
        document,
        old_child_tag,
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
