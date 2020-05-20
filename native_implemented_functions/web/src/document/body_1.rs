//! ```elixir
//! case Lumen.Web.Document.body(document) do
//!   {:ok, body} -> ...
//!   :error -> ...
//! end
//! ```

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::document::document_from_term;
use crate::option_to_ok_tuple_or_error;

use native_implemented_function::native_implemented_function;

#[native_implemented_function(body/1)]
pub fn result(process: &Process, document: Term) -> exception::Result<Term> {
    let document_document = document_from_term(document)?;

    option_to_ok_tuple_or_error(process, document_document.body()).map_err(|error| error.into())
}
