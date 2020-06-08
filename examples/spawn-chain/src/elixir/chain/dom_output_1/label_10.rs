//! ```elixir
//! # label 10
//! # pushed to stack: (document, tr, text_td)
//! # returned from call: :ok
//! # full stack: (:ok, document, tr, text_td)
//! # returns: :ok
//! Lumen::Web::Node.append_child(tr, text_td)
//!
//! {:ok, tbody} = Lumen::Web::Document.get_element_by_id(document, "output")
//! Lumen::Web::Node.append_child(tbody, tr)
//! ```

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use super::label_11;

// Private

#[native_implemented::label]
fn result(process: &Process, ok: Term, document: Term, tr: Term, text_td: Term) -> Term {
    assert_eq!(ok, Atom::str_to_term("ok"));
    assert!(document.is_boxed_resource_reference());
    assert!(tr.is_boxed_resource_reference());
    assert!(text_td.is_boxed_resource_reference());

    process.queue_frame_with_arguments(
        liblumen_web::node::append_child_2::frame().with_arguments(false, &[tr, text_td]),
    );
    process.queue_frame_with_arguments(label_11::frame().with_arguments(true, &[document, tr]));

    Term::NONE
}
