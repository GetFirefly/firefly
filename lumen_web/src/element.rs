pub mod class_name_1;
pub mod remove_1;
pub mod set_attribute_3;

use std::convert::TryInto;
use std::mem;

use anyhow::*;

use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use wasm_bindgen::JsCast;

use web_sys::{Element, HtmlBodyElement, HtmlElement, HtmlTableElement, Node};

pub fn module() -> Atom {
    Atom::try_from_str("Elixir.Lumen.Web.Element").unwrap()
}

// Private

fn from_term(term: Term) -> Result<&'static Element, exception::Exception> {
    let boxed: Boxed<Resource> = term
        .try_into()
        .with_context(|| format!("{} is not a resource", term))?;
    let resource_reference: Resource = boxed.into();

    if resource_reference.is::<Element>() {
        let element: &Element = resource_reference.downcast_ref().unwrap();
        let static_element: &'static Element =
            unsafe { mem::transmute::<&Element, &'static Element>(element) };

        Ok(static_element)
    } else if resource_reference.is::<HtmlBodyElement>() {
        let html_body_element: &HtmlBodyElement = resource_reference.downcast_ref().unwrap();
        let static_element: &'static Element =
            unsafe { mem::transmute::<&Element, &'static Element>(html_body_element.as_ref()) };

        Ok(static_element)
    } else if resource_reference.is::<HtmlElement>() {
        let html_element: &HtmlElement = resource_reference.downcast_ref().unwrap();
        let static_element: &'static Element =
            unsafe { mem::transmute::<&Element, &'static Element>(html_element.as_ref()) };

        Ok(static_element)
    } else if resource_reference.is::<HtmlTableElement>() {
        let html_table_element: &HtmlTableElement = resource_reference.downcast_ref().unwrap();
        let static_element: &'static Element =
            unsafe { mem::transmute::<&Element, &'static Element>(html_table_element.as_ref()) };

        Ok(static_element)
    } else if resource_reference.is::<Node>() {
        let node: &Node = resource_reference.downcast_ref().unwrap();
        match node.dyn_ref() {
            Some(element) => {
                let static_element: &'static Element =
                    unsafe { mem::transmute::<&Element, &'static Element>(element) };

                Ok(static_element)
            }
            None => Err(badarg!(process).into()),
        }
    } else {
        Err(badarg!(process).into())
    }
}
