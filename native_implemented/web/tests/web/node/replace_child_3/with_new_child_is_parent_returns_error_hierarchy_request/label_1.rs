//! ```elixir
//! # label 1
//! # pushed to stack: ("div")
//! # returned form call: {:ok, document}
//! # full stack: ({:ok, document}, "div")
//! # returns: {:ok, old_child}
//! {:ok, old_child} = Lumen.Web.Document.create_element(document, "table")
//! {:ok, parent} = Lumen.Web.Document.create_element(document, "div")
//! :ok = Lumen.Web.Node.append_child(parent, old_child)
//! {:error, :hierarchy_request} = Lumen.Web.replace_child(parent, old_child, parent)
//! ! ```

use std::convert::TryInto;

use liblumen_alloc::erts::exception;
use liblumen_rt::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use super::label_2;

#[native_implemented::label]
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

    let old_child_tag = process.binary_from_str("table");
    process.queue_frame_with_arguments(
        liblumen_web::document::create_element_2::frame()
            .with_arguments(false, &[document, old_child_tag]),
    );

    process.queue_frame_with_arguments(label_2::frame().with_arguments(true, &[document]));

    Ok(Term::NONE)
}
