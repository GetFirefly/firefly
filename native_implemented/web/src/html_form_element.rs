pub mod element_2;

use std::convert::TryInto;
use std::mem;

use anyhow::*;
use wasm_bindgen::JsCast;
use web_sys::{EventTarget, HtmlFormElement};

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::term::prelude::*;

// Private

fn from_term(term: Term) -> Result<&'static HtmlFormElement, exception::Exception> {
    let boxed: Boxed<Resource> = term
        .try_into()
        .with_context(|| format!("{} must be an HTML form element resource", term))?;
    let resource_reference: Resource = boxed.into();

    if resource_reference.is::<EventTarget>() {
        let event_target: &EventTarget = resource_reference.downcast_ref().unwrap();

        if let Some(html_form_element) = event_target.dyn_ref() {
            let static_html_form_element: &'static HtmlFormElement = unsafe {
                mem::transmute::<&HtmlFormElement, &'static HtmlFormElement>(html_form_element)
            };

            Ok(static_html_form_element)
        } else {
            Err(TypeError)
                .with_context(|| {
                    format!(
                        "{} is an event target resource, but not an HTML form element",
                        term
                    )
                })
                .map_err(From::from)
        }
    } else {
        Err(TypeError)
            .with_context(|| format!("{} is a resource, but not an event target", term))
            .map_err(From::from)
    }
}

fn module() -> Atom {
    Atom::try_from_str("Elixir.Lumen.Web.HTMLFormElement").unwrap()
}

fn module_id() -> usize {
    module().id()
}
