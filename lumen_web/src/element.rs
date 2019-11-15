pub mod class_name_1;
pub mod remove_1;
pub mod set_attribute_3;

use std::any::TypeId;
use std::convert::TryInto;
use std::mem;

use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::term::prelude::*;

use wasm_bindgen::JsCast;

use web_sys::{Element, HtmlBodyElement, HtmlElement, HtmlTableElement, Node};

pub fn module() -> Atom {
    Atom::try_from_str("Elixir.Lumen.Web.Element").unwrap()
}

// Private

fn from_term(term: Term) -> Result<&'static Element, exception::Exception> {
    let boxed: Boxed<Resource> = term.try_into()?;
    let resource_reference: Resource = boxed.into();

    let resource_type_id = resource_reference.type_id();

    if resource_type_id == TypeId::of::<Element>() {
        let element: &Element = resource_reference.downcast_ref().unwrap();
        let static_element: &'static Element =
            unsafe { mem::transmute::<&Element, &'static Element>(element) };

        Ok(static_element)
    } else if resource_type_id == TypeId::of::<HtmlBodyElement>() {
        let html_body_element: &HtmlBodyElement = resource_reference.downcast_ref().unwrap();
        let static_element: &'static Element =
            unsafe { mem::transmute::<&Element, &'static Element>(html_body_element.as_ref()) };

        Ok(static_element)
    } else if resource_type_id == TypeId::of::<HtmlElement>() {
        let html_element: &HtmlElement = resource_reference.downcast_ref().unwrap();
        let static_element: &'static Element =
            unsafe { mem::transmute::<&Element, &'static Element>(html_element.as_ref()) };

        Ok(static_element)
    } else if resource_type_id == TypeId::of::<HtmlTableElement>() {
        let html_table_element: &HtmlTableElement = resource_reference.downcast_ref().unwrap();
        let static_element: &'static Element =
            unsafe { mem::transmute::<&Element, &'static Element>(html_table_element.as_ref()) };

        Ok(static_element)
    } else if resource_type_id == TypeId::of::<Node>() {
        let node: &Node = resource_reference.downcast_ref().unwrap();
        match node.dyn_ref() {
            Some(element) => {
                let static_element: &'static Element =
                    unsafe { mem::transmute::<&Element, &'static Element>(element) };

                Ok(static_element)
            }
            None => Err(badarg!().into()),
        }
    } else {
        Err(badarg!().into())
    }
}
