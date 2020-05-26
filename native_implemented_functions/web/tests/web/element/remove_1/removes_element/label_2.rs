//! ```elixir
//! # label 2
//! # pushed to stack: ()
//! # returned from call: {:ok, document}
//! # full stack: ({:ok, document})
//! # returns: {:ok, body}
//! {:ok, body} = Lumen.Web.Document.body(document)
//! {:ok, child} = Lumen.Web.Document.create_element(body, "table");
//! :ok = Lumen.Web.Node.append_child(document, child);
//! :ok = Lumen.Web.Element.remove(child);
//! Lumen.Web.Wait.with_return(body_tuple)
//! ```

use std::convert::TryInto;

use web_sys::Document;

use liblumen_alloc::erts::process::{Frame, Native, Process};
use liblumen_alloc::erts::term::prelude::*;

use liblumen_web::runtime::process::current_process;

use super::label_3;

pub fn frame() -> Frame {
    super::frame(NATIVE)
}

// Private

const NATIVE: Native = Native::One(native);

extern "C" fn native(ok_document: Term) -> Term {
    let arc_process = current_process();
    arc_process.reduce();

    result(&arc_process, ok_document)
}

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
    let document_ref_boxed: Boxed<Resource> = document.try_into().unwrap();
    let document_reference: Resource = document_ref_boxed.into();
    let _: &Document = document_reference.downcast_ref().unwrap();

    process.queue_frame_with_arguments(label_3::frame().with_arguments(true, &[document]));
    process.queue_frame_with_arguments(
        liblumen_web::document::body_1::frame().with_arguments(false, &[document]),
    );

    Term::NONE
}
