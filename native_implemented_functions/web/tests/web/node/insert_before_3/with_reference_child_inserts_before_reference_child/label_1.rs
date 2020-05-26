//! ```elixir
//! # label 1
//! # pushed to stack: ()
//! # returned form call: {:ok, document}
//! # full stack: ({:ok, document})
//! # returns: {:ok, reference_child}
//! {:ok, reference_child} = Lumen.Web.Document.create_element(document, "table")
//! {:ok, parent} = Lumen.Web.Document.create_element(parent_document, "div")
//! :ok = Lumen.Web.Node.append_child(document, parent)
//! :ok = Lumen.Web.Node.append_child(parent, reference_child)
//! {:ok, new_child} = Lumen.Web.Document.create_element(document, "ul");
//! {:ok, inserted_child} = Lumen.Web.insert_before(parent, new_child, reference_child)
//! ```

use std::convert::TryInto;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::{Frame, Native, Process};
use liblumen_alloc::erts::term::prelude::*;

use liblumen_web::runtime::process::current_process;

use super::label_2;

pub fn frame() -> Frame {
    super::frame(NATIVE)
}

// Private

const NATIVE: Native = Native::One(native);

extern "C" fn native(ok_document: Term) -> Term {
    let arc_process = current_process();
    arc_process.reduce();

    arc_process.return_status(result(&arc_process, ok_document))
}

fn result(process: &Process, ok_document: Term) -> exception::Result<Term> {
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

    process.queue_frame_with_arguments(label_2::frame().with_arguments(true, &[document]));

    let reference_child_tag = process.binary_from_str("table")?;
    process.queue_frame_with_arguments(
        liblumen_web::document::create_element_2::frame()
            .with_arguments(false, &[document, reference_child_tag]),
    );

    Ok(Term::NONE)
}
