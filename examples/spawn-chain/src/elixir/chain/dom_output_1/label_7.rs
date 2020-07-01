//! ```elixir
//! # label 7
//! # pushed to stack: (document, tr, text)
//! # returned from call: :ok
//! # full stack: (:ok, document, tr, text)
//! # returns: text_text
//! {:ok, text_text} = Lumen::Web::Document.create_text_node(document, text)
//! {:ok, text_td} = Lumen::Web::Document.create_element(document, "td")
//! Lumen::Web::Node.append_child(text_td, text_text)
//! Lumen::Web::Node.append_child(tr, text_td)
//!
//! {:ok, tbody} = Lumen::Web::Document.get_element_by_id(document, "output")
//! Lumen::Web::Node.append_child(tbody, tr)
//! ```

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use super::label_8;

// Private

#[native_implemented::label]
fn result(process: &Process, ok: Term, document: Term, tr: Term, text: Term) -> Term {
    assert_eq!(ok, Atom::str_to_term("ok"));
    assert!(document.is_boxed_resource_reference());
    assert!(tr.is_boxed_resource_reference());

    process.queue_frame_with_arguments(
        liblumen_web::document::create_text_node_2::frame()
            .with_arguments(false, &[document, text]),
    );
    process.queue_frame_with_arguments(label_8::frame().with_arguments(true, &[document, tr]));

    Term::NONE
}
