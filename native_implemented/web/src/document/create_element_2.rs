//! ```elixir
//! case Lumen.Web.Document.create_element(document, tag) do
//!   {:ok, element} -> ...
//!   {:error, {:tag, tag}} -> ...
//! end
//! ```

use liblumen_alloc::atom;
use liblumen_alloc::erts::exception;
use liblumen_rt::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::runtime::binary_to_string::binary_to_string;
use crate::{document, ok_tuple};

#[native_implemented::function(Elixir.Lumen.Web.Document:create_element/2)]
pub fn result(process: &Process, document: Term, tag: Term) -> exception::Result<Term> {
    let document_document = document::from_term(document)?;
    let tag_string: String = binary_to_string(tag)?;

    let final_term = match document_document.create_element(&tag_string) {
        Ok(element) => ok_tuple(process, element),
        Err(_) => {
            let tag_tag = Atom::str_to_term("tag");
            let reason = process.tuple_from_slice(&[tag_tag, tag]);

            let error = atom!("error");

            process.tuple_from_slice(&[error, reason])
        }
    };

    Ok(final_term)
}
