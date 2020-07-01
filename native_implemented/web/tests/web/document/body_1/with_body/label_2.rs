//! ```elixir
//! # label 2
//! # pushed to stack: ()
//! # returned from call: {:ok, document}
//! # full stack: ({:ok, document})
//! # returns: {:ok, body} | :error
//! body_tuple = Lumen.Web.Document.body(document)
//! Lumen.Web:.Wait.with_return(body_tuple)
//! ```

use std::convert::TryInto;

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

#[native_implemented::label]
fn result(process: &Process, ok_document: Term) -> Term {
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

    process.queue_frame_with_arguments(
        liblumen_web::document::body_1::frame().with_arguments(false, &[document]),
    );

    Term::NONE
}
