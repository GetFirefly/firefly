//! ```elixir
//! case Lumen.Web.Document.get_element_by_id(document, "element-id") do
//!   {:ok, element} -> ...
//!   :error -> ...
//! end
//! ```

use liblumen_alloc::erts::exception;
use liblumen_rt::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::runtime::binary_to_string::binary_to_string;
use crate::{document, option_to_ok_tuple_or_error};

#[native_implemented::function(Elixir.Lumen.Web.Document:get_element_by_id/2)]
pub fn result(process: &Process, document: Term, id: Term) -> exception::Result<Term> {
    let document_document = document::from_term(document)?;
    let id_string: String = binary_to_string(id)?;

    Ok(option_to_ok_tuple_or_error(
        process,
        document_document.get_element_by_id(&id_string),
    ))
}
