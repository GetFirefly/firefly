/// Node is an interface from which various types of DOM API objects inherit. This allows these
/// types to be treated similarly; for example, inheriting the same set of methods or being
/// tested in the same way.
pub mod append_child_2;
pub mod insert_before_3;
pub mod replace_child_3;

use std::convert::TryInto;
use std::mem;

use anyhow::*;
use web_sys::{Document, Element, HtmlBodyElement, HtmlElement, HtmlTableElement, Node, Text};

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::term::prelude::*;

fn module() -> Atom {
    Atom::try_from_str("Elixir.Lumen.Web.Node").unwrap()
}

fn module_id() -> usize {
    module().id()
}

fn node_from_term(term: Term) -> Result<&'static Node, exception::Exception> {
    let boxed: Boxed<Resource> = term
        .try_into()
        .with_context(|| format!("{} must be a source", term))?;
    let resource_reference: Resource = boxed.into();

    if resource_reference.is::<Document>() {
        let document: &Document = resource_reference.downcast_ref().unwrap();
        let node: &'static Node =
            unsafe { mem::transmute::<&Node, &'static Node>(document.as_ref()) };

        Ok(node)
    } else if resource_reference.is::<Element>() {
        let element: &Element = resource_reference.downcast_ref().unwrap();
        let node: &'static Node =
            unsafe { mem::transmute::<&Node, &'static Node>(element.as_ref()) };

        Ok(node)
    } else if resource_reference.is::<HtmlBodyElement>() {
        let html_body_element: &HtmlBodyElement = resource_reference.downcast_ref().unwrap();
        let node: &'static Node =
            unsafe { mem::transmute::<&Node, &'static Node>(html_body_element.as_ref()) };

        Ok(node)
    } else if resource_reference.is::<HtmlElement>() {
        let html_element: &HtmlElement = resource_reference.downcast_ref().unwrap();
        let node: &'static Node =
            unsafe { mem::transmute::<&Node, &'static Node>(html_element.as_ref()) };

        Ok(node)
    } else if resource_reference.is::<HtmlTableElement>() {
        let html_table_element: &HtmlTableElement = resource_reference.downcast_ref().unwrap();
        let node: &'static Node =
            unsafe { mem::transmute::<&Node, &'static Node>(html_table_element.as_ref()) };

        Ok(node)
    } else if resource_reference.is::<Node>() {
        let node: &Node = resource_reference.downcast_ref().unwrap();
        let static_node: &'static Node = unsafe { mem::transmute::<&Node, &'static Node>(node) };

        Ok(static_node)
    } else if resource_reference.is::<Text>() {
        let text: &Text = resource_reference.downcast_ref().unwrap();
        let node: &'static Node = unsafe { mem::transmute::<&Node, &'static Node>(text.as_ref()) };

        Ok(node)
    } else {
        Err(TypeError)
            .with_context(|| format!("{} is a resource, but cannot be converted to a node", term))
            .map_err(From::from)
    }
}
