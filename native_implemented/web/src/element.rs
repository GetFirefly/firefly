pub mod class_name_1;
pub mod remove_1;
pub mod set_attribute_3;

use std::convert::TryInto;
use std::mem;

use anyhow::*;
use wasm_bindgen::JsCast;
use web_sys::{Element, HtmlBodyElement, HtmlElement, HtmlTableElement, Node};

use liblumen_alloc::erts::exception::InternalResult;
use liblumen_alloc::erts::term::prelude::*;

pub fn module() -> Atom {
    Atom::try_from_str("Elixir.Lumen.Web.Element").unwrap()
}

fn module_id() -> usize {
    module().id()
}

// Private

fn from_term(term: Term) -> InternalResult<&'static Element> {
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
            None => Err(anyhow!("{} is a Node, but not an Element", term).into()),
        }
    } else {
        Err(anyhow!(
            "{} is a resource, but not an Element, HTMLBodyElement, HTMLElement, HTMLTableElement, or Node", term
        ).into())
    }
}
