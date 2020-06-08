//! ```elixir
//! # label 1
//! # pushed to stack: (text)
//! # returned from call: {:ok, window}
//! # full stack: ({:ok, window}, text)
//! {:ok, document} = Lumen::Web::Window.document(window)
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

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use super::label_2;

// Private

#[native_implemented::label]
fn result(process: &Process, ok_window: Term, text: Term) -> Term {
    assert!(
        ok_window.is_boxed_tuple(),
        "ok_window ({:?}) is not a tuple",
        ok_window
    );
    let ok_window_tuple: Boxed<Tuple> = ok_window.try_into().unwrap();
    assert_eq!(ok_window_tuple.len(), 2);
    assert_eq!(ok_window_tuple[0], Atom::str_to_term("ok"));
    let window = ok_window_tuple[1];
    assert!(window.is_boxed_resource_reference());

    assert!(text.is_binary());

    process.queue_frame_with_arguments(
        liblumen_web::window::document_1::frame().with_arguments(false, &[window]),
    );
    process.queue_frame_with_arguments(label_2::frame().with_arguments(true, &[text]));

    Term::NONE
}
