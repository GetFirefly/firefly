pub mod value_1;

use std::convert::TryInto;
use std::mem;

use anyhow::*;
use web_sys::HtmlInputElement;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::term::prelude::*;

// Private

fn from_term(term: Term) -> Result<&'static HtmlInputElement, exception::Exception> {
    let boxed: Boxed<Resource> = term
        .try_into()
        .with_context(|| format!("{} must be HTML input element resource", term))?;
    let html_input_element_reference: Resource = boxed.into();

    match html_input_element_reference.downcast_ref() {
        Some(html_input_element) => {
            let static_html_input_element: &'static HtmlInputElement =
                unsafe { mem::transmute::<&HtmlInputElement, _>(html_input_element) };

            Ok(static_html_input_element)
        }
        None => Err(TypeError)
            .with_context(|| format!("{} is a resource, but not an HTML input element", term))
            .map_err(From::from),
    }
}

fn module() -> Atom {
    Atom::try_from_str("Elixir.Lumen.Web.HTMLInputElement").unwrap()
}

fn module_id() -> usize {
    module().id()
}
