pub mod value_1;

use std::convert::TryInto;
use std::mem;

use web_sys::HtmlInputElement;

use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::term::{resource, Atom, Term};

// Private

fn from_term(term: Term) -> Result<&'static HtmlInputElement, exception::Exception> {
    let html_input_element_reference: resource::Reference = term.try_into()?;

    match html_input_element_reference.downcast_ref() {
        Some(html_input_element) => {
            let static_html_input_element: &'static HtmlInputElement =
                unsafe { mem::transmute::<&HtmlInputElement, _>(html_input_element) };

            Ok(static_html_input_element)
        }
        None => Err(badarg!().into()),
    }
}

fn module() -> Atom {
    Atom::try_from_str("Elixir.Lumen.Web.HTMLInputElement").unwrap()
}
