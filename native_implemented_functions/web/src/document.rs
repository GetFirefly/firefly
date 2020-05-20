pub mod body_1;
/// The Document interface represents any web page loaded in the browser
pub mod create_element_2;
pub mod create_text_node_2;
pub mod get_element_by_id_2;
pub mod new_0;

use std::convert::TryInto;
use std::mem;

use anyhow::*;
use web_sys::Document;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::term::prelude::*;

// Private

fn module() -> Atom {
    Atom::try_from_str("Elixir.Lumen.Web.Document").unwrap()
}

fn module_id() -> usize {
    module().id()
}

fn document_from_term(term: Term) -> Result<&'static Document, exception::Exception> {
    let boxed: Boxed<Resource> = term
        .try_into()
        .with_context(|| format!("{} must be a document resource", term))?;
    let document_reference: Resource = boxed.into();

    match document_reference.downcast_ref() {
        Some(document) => {
            let static_document: &'static Document =
                unsafe { mem::transmute::<&Document, _>(document) };

            Ok(static_document)
        }
        None => Err(TypeError)
            .with_context(|| format!("{} is a resource, but not a document", term))
            .map_err(From::from),
    }
}
