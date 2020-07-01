//! ```elixir
//! # label 4
//! # pushed to stack: (document, tr, text)
//! # returned from call: pid_text
//! # full stack: (pid_text, document, tr, text)
//! # returns: {:ok, pid_td}
//! {:ok, pid_td} = Lumen::Web::Document.create_element(document, "td")
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

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use super::label_5;

// Private

#[native_implemented::label]
fn result(
    process: &Process,
    pid_text: Term,
    document: Term,
    tr: Term,
    text: Term,
) -> exception::Result<Term> {
    let _: Boxed<Resource> = pid_text.try_into().unwrap();
    let _: Boxed<Resource> = document.try_into().unwrap();
    let _: Boxed<Resource> = tr.try_into().unwrap();

    let tag = process.binary_from_str("td")?;
    process.queue_frame_with_arguments(
        liblumen_web::document::create_element_2::frame().with_arguments(false, &[document, tag]),
    );

    process.queue_frame_with_arguments(
        label_5::frame().with_arguments(true, &[document, tr, pid_text, text]),
    );

    Ok(Term::NONE)
}
