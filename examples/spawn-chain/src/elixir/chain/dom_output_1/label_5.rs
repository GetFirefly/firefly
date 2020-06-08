//! ```elixir
//! # label 5
//! # pushed to stack: (document, tr, pid_text, text)
//! # returned from call: {:ok, pid_td}
//! # full stack: ({:ok, pid_td}, document, tr, pid_text, text)
//! # returns: :ok
//! Lumen::Web::Node.append_child(pid_td, pid_text)
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

use std::convert::TryInto;

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use super::label_6;

// Private

#[native_implemented::label]
fn result(
    process: &Process,
    ok_pid_td: Term,
    document: Term,
    tr: Term,
    pid_text: Term,
    text: Term,
) -> Term {
    assert!(
        ok_pid_td.is_boxed_tuple(),
        "ok_pid_td ({}) is not a tuple",
        ok_pid_td
    );
    assert!(document.is_boxed_resource_reference());
    assert!(tr.is_boxed_resource_reference());
    assert!(pid_text.is_boxed_resource_reference());

    let ok_pid_td_tuple: Boxed<Tuple> = ok_pid_td.try_into().unwrap();
    assert_eq!(ok_pid_td_tuple.len(), 2);
    assert_eq!(ok_pid_td_tuple[0], Atom::str_to_term("ok"));
    let pid_td = ok_pid_td_tuple[1];
    assert!(pid_td.is_boxed_resource_reference());

    process.queue_frame_with_arguments(
        liblumen_web::node::append_child_2::frame().with_arguments(false, &[pid_td, pid_text]),
    );

    process.queue_frame_with_arguments(
        label_6::frame().with_arguments(true, &[document, tr, pid_td, text]),
    );

    Term::NONE
}
