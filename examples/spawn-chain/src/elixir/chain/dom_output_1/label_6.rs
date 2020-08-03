//! ```elixir
//! # label 6
//! # pushed to stack: (document, tr, pid_td, text)
//! # returned from call: :ok
//! # full stack: (:ok, document, tr, pid_td, text)
//! # returns: :ok
//! Lumen::Web::Node.append_child(tr, pid_td)
//!
//! {:ok, text_text} = Lumen::Web::Document.create_text_node(document, to_string(text))
//! {:ok, text_td} = Lumen::Web::Document.create_element(document, "td")
//! Lumen::Web::Node.append_child(text_td, text_text)
//! Lumen::Web::Node.append_child(tr, text_td)
//!
//! {:ok, tbody} = Lumen::Web::Document.get_element_by_id(document, "output")
//! Lumen::Web::Node.append_child(tbody, tr)
//! ```

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use super::label_7;

// Private

#[native_implemented::label]
fn result(process: &Process, ok: Term, document: Term, tr: Term, pid_td: Term, text: Term) -> Term {
    assert_eq!(ok, Atom::str_to_term("ok"));
    assert!(document.is_boxed_resource_reference());
    assert!(tr.is_boxed_resource_reference());
    assert!(pid_td.is_boxed_resource_reference());

    process.queue_frame_with_arguments(
        liblumen_web::node::append_child_2::frame().with_arguments(false, &[tr, pid_td]),
    );
    process
        .queue_frame_with_arguments(label_7::frame().with_arguments(true, &[document, tr, text]));

    Term::NONE
}
