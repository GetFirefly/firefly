use std::convert::TryInto;
use std::sync::Arc;

use liblumen_alloc::erts::process::frames::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::{frames, Process};
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::ModuleFunctionArity;

use web_sys::Document;

use super::label_3;

pub fn place_frame(process: &Process, placement: Placement) {
    process.place_frame(frame(), placement);
}

// Private

// ```elixir
// # label 2
// # pushed to stack: ()
// # returned from call: {:ok, document}
// # full stack: ({:ok, document})
// # returns: {:ok, body}
// {:ok, body} = Lumen.Web.Document.body(document)
// class_name = Lumen.Web.Element.class_name(body)
// Lumen.Web.Wait.with_return(class_name)
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
    let document_ref_boxed: Boxed<Resource> = document.try_into().unwrap();
    let document_reference: Resource = document_ref_boxed.into();
    let _: &Document = document_reference.downcast_ref().unwrap();

    label_3::place_frame(arc_process, Placement::Replace);
    liblumen_web::document::body_1::place_frame_with_arguments(
        arc_process,
        Placement::Push,
        document,
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
