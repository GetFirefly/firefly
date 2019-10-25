pub mod element_2;

use std::any::TypeId;
use std::convert::TryInto;
use std::mem;

use wasm_bindgen::JsCast;

use web_sys::{EventTarget, HtmlFormElement};

use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::term::prelude::*;

// Private

fn from_term(term: Term) -> Result<&'static HtmlFormElement, exception::Exception> {
    let resource_reference: resource::Reference = term.try_into()?;

    let resource_type_id = resource_reference.type_id();

    if resource_type_id == TypeId::of::<EventTarget>() {
        let event_target: &EventTarget = resource_reference.downcast_ref().unwrap();

        if let Some(html_form_element) = event_target.dyn_ref() {
            let static_html_form_element: &'static HtmlFormElement = unsafe {
                mem::transmute::<&HtmlFormElement, &'static HtmlFormElement>(html_form_element)
            };

            Ok(static_html_form_element)
        } else {
            Err(badarg!().into())
        }
    } else {
        Err(badarg!().into())
    }
}

fn module() -> Atom {
    Atom::try_from_str("Elixir.Lumen.Web.HTMLFormElement").unwrap()
}
