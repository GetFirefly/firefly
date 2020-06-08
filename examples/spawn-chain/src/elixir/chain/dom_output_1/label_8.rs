//! ```elixir
//! # label 8
//! # pushed to stack: (document, tr)
//! # returned from call: text_text
//! # full stack: (text_text, document, tr)
//! # returns: {:ok, text_td}
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

use super::label_9;

// Private

#[native_implemented::label]
fn result(process: &Process, text_text: Term, document: Term, tr: Term) -> exception::Result<Term> {
    let _: Boxed<Resource> = text_text.try_into().unwrap();
    let _: Boxed<Resource> = document.try_into().unwrap();
    let _: Boxed<Resource> = tr.try_into().unwrap();

    let tag = process.binary_from_str("td").unwrap();
    process.queue_frame_with_arguments(
        liblumen_web::document::create_element_2::frame().with_arguments(false, &[document, tag]),
    );

    process.queue_frame_with_arguments(
        label_9::frame().with_arguments(true, &[document, tr, text_text]),
    );

    Ok(Term::NONE)
}
