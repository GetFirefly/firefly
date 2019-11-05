//! ```elixir
//! case Lumen.Web.Document.create_element(document, tag) do
//!   {:ok, element} -> ...
//!   {:error, {:tag, tag}} -> ...
//! end
//! ```

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::{atom_unchecked, Term};

use lumen_runtime::binary_to_string::binary_to_string;

use crate::document::document_from_term;
use crate::{error, ok_tuple};

use lumen_runtime_macros::native_implemented_function;

#[native_implemented_function(create_element/2)]
pub fn native(process: &Process, document: Term, tag: Term) -> exception::Result {
    let document_document = document_from_term(document)?;
    let tag_string: String = binary_to_string(tag)?;

    match document_document.create_element(&tag_string) {
        Ok(element) => ok_tuple(process, Box::new(element)),
        Err(_) => {
            let tag_tag = atom_unchecked("tag");
            let reason = process.tuple_from_slice(&[tag_tag, tag])?;

            let error = error();

            process.tuple_from_slice(&[error, reason])
        }
    }
    .map_err(|error| error.into())
}
