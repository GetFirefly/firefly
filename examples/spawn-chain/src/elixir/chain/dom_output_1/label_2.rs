//! ```elixir
//! # label 2
//! # pushed to stack: (text)
//! # returned from call: {:ok, document}
//! # full stack: ({:ok, document}, text)
//! {:ok, tr} = Lumen::Web::Document.create_element(document, "tr")
//!
//! {:ok, pid_text} = Lumen::Web::Document.create_text_node(document, to_string(self()))
//! {:ok, pid_td} = Lumen::Web::Document.create_element(document, "td")
//! Lumen::Web::Node.append_child(pid_td, pid_text);
//! Lumen::Web::Node.append_child(tr, pid_td)
//!
//! {:ok, text_text} = Lumen::Web::Document.create_text_node(document, to_string(text()))
//! {:ok, text_td} = Lumen::Web::Document.create_element(document, "td")
//! Lumen::Web::Node.append_child(text_td, text_text);
//! Lumen::Web::Node.append_child(tr, text_td)
//!
//! {:ok, tbody} = Lumen::Web::Document.get_element_by_id(document, "output")
//! Lumen::Web::Node.append_child(tbody, tr)
//! ```

use std::convert::TryInto;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use super::label_3;

// Private

#[native_implemented::label]
fn result(process: &Process, ok_document: Term, text: Term) -> exception::Result<Term> {
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

    let tag = process.binary_from_str("tr")?;
    process.queue_frame_with_arguments(
        liblumen_web::document::create_element_2::frame().with_arguments(false, &[document, tag]),
    );

    process.queue_frame_with_arguments(label_3::frame().with_arguments(true, &[document, text]));

    Ok(Term::NONE)
}
