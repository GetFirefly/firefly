//! ```elixir
//! # label 3
//! # pushed to stack: (document, text)
//! # returned from call: {:ok, tr}
//! # full stack: ({:ok, tr}, document, text)
//! # returns: {:ok, pid_text}
//! {:ok, pid_text} = Lumen::Web::Document.create_text_node(document, to_string(self()))
//! {:ok, pid_td} = Lumen::Web::Document.create_element(document, "td")
//! Lumen::Web::Node.append_child(pid_td, pid_text)
//! Lumen::Web::Node.append_child(tr, pid_td)
//!
//! {:ok, text_text} = Lumen::Web::Document.create_text_node(document, to_string(text()))
//! {:ok, text_td} = Lumen::Web::Document.create_element(document, "td")
//! Lumen::Web::Node.append_child(text_td, text_text)
//! Lumen::Web::Node.append_child(tr, text_td)
//!
//! {:ok, tbody} = Lumen::Web::Document.get_element_by_id(document, "output")
//! Lumen::Web::Node.append_child(tbody, tr)
//! ```

use std::convert::TryInto;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use super::label_4;

// Private

#[native_implemented::label]
fn result(process: &Process, ok_tr: Term, document: Term, text: Term) -> exception::Result<Term> {
    assert!(ok_tr.is_boxed_tuple());
    assert!(document.is_boxed_resource_reference());

    let ok_tr_tuple: Boxed<Tuple> = ok_tr.try_into().unwrap();
    assert_eq!(ok_tr_tuple.len(), 2);
    assert_eq!(ok_tr_tuple[0], Atom::str_to_term("ok"));
    let tr = ok_tr_tuple[1];
    assert!(tr.is_boxed_resource_reference());

    // TODO actually call `to_string(self)`
    let pid_text_binary = process.binary_from_str(&format!("{}", process.pid_term()))?;
    process.queue_frame_with_arguments(
        liblumen_web::document::create_text_node_2::frame()
            .with_arguments(false, &[document, pid_text_binary]),
    );

    process
        .queue_frame_with_arguments(label_4::frame().with_arguments(true, &[document, tr, text]));

    Ok(Term::NONE)
}
