//! ```elixir
//! # label 9
//! # pushed to stack: (document, tr, text_text)
//! # returned from call: {:ok, text_td}
//! # full stack: ({:ok, text_td}, document, tr, text_text)
//! # returns: :ok
//! Lumen::Web::Node.append_child(text_td, text_text)
//! Lumen::Web::Node.append_child(tr, text_td)
//!
//! {:ok, tbody} = Lumen::Web::Document.get_element_by_id(document, "output")
//! Lumen::Web::Node.append_child(tbody, tr)
//! ```

use std::convert::TryInto;

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use super::label_10;

// Private

#[native_implemented::label]
fn result(process: &Process, ok_text_td: Term, document: Term, tr: Term, text_text: Term) -> Term {
    assert!(ok_text_td.is_boxed_tuple());
    assert!(document.is_boxed_resource_reference());
    assert!(tr.is_boxed_resource_reference());
    assert!(text_text.is_boxed_resource_reference());

    let ok_text_td_tuple: Boxed<Tuple> = ok_text_td.try_into().unwrap();
    assert_eq!(ok_text_td_tuple.len(), 2);
    assert_eq!(ok_text_td_tuple[0], Atom::str_to_term("ok"));
    let text_td = ok_text_td_tuple[1];
    assert!(text_td.is_boxed_resource_reference());

    process.queue_frame_with_arguments(
        liblumen_web::node::append_child_2::frame().with_arguments(false, &[text_td, text_text]),
    );
    process.queue_frame_with_arguments(
        label_10::frame().with_arguments(true, &[document, tr, text_td]),
    );

    Term::NONE
}
