//! ```elixir
//! # label 11
//! # pushed to stack: (document, tr)
//! # returned from call: :ok
//! # full stack: (:ok, document, tr)
//! # returns: {:ok, tbody}
//! {:ok, tbody} = Lumen::Web::Document.get_element_by_id(document, "output")
//! Lumen::Web::Node.append_child(tbody, tr)
//! ```

use liblumen_alloc::atom;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use super::label_12;

// Private

#[native_implemented::label]
fn result(process: &Process, ok: Term, document: Term, tr: Term) -> exception::Result<Term> {
    assert_eq!(ok, atom!("ok"));

    let id = process.binary_from_str("output")?;
    process.queue_frame_with_arguments(
        liblumen_web::document::get_element_by_id_2::frame().with_arguments(false, &[document, id]),
    );

    process.queue_frame_with_arguments(label_12::frame().with_arguments(true, &[tr]));

    Ok(Term::NONE)
}
