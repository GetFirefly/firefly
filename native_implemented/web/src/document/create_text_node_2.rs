//! ```elixir
//! text = Lumen.Web.Document.create_text_node(document, data)
//! ```

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::runtime::binary_to_string::binary_to_string;

use crate::document;

#[native_implemented::function(create_text_node/2)]
pub fn result(process: &Process, document: Term, data: Term) -> exception::Result<Term> {
    let document_document = document::from_term(document)?;
    let data_string: String = binary_to_string(data)?;

    let text = document_document.create_text_node(&data_string);

    process.resource(text).map_err(|error| error.into())
}
