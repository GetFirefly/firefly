//! ```elixir
//! text = Lumen.Web.Document.create_text_node(document, data)
//! ```

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use native_implemented_function::native_implemented_function;

use lumen_rt_full::binary_to_string::binary_to_string;

use crate::document::document_from_term;

#[native_implemented_function(create_text_node/2)]
pub fn result(process: &Process, document: Term, data: Term) -> exception::Result<Term> {
    let document_document = document_from_term(document)?;
    let data_string: String = binary_to_string(data)?;

    let text = document_document.create_text_node(&data_string);
    let text_box = Box::new(text);

    process.resource(text_box).map_err(|error| error.into())
}
