/// Node is an interface from which various types of DOM API objects inherit. This allows these
/// types to be treated similarly; for example, inheriting the same set of methods or being
/// tested in the same way.
pub mod append_child_2;

use std::any::TypeId;
use std::convert::TryInto;
use std::mem;

use web_sys::{Document, Element, HtmlElement, Node, Text};

use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::term::{resource, Atom, Term};

fn module() -> Atom {
    Atom::try_from_str("Lumen.Web.Node").unwrap()
}

fn node_from_term(term: Term) -> Result<&'static Node, exception::Exception> {
    let resource_reference: resource::Reference = term.try_into()?;

    let resource_type_id = resource_reference.type_id();

    if resource_type_id == TypeId::of::<Document>() {
        let document: &Document = resource_reference.downcast_ref().unwrap();
        let node: &'static Node =
            unsafe { mem::transmute::<&Node, &'static Node>(document.as_ref()) };

        Ok(node)
    } else if resource_type_id == TypeId::of::<Element>() {
        let element: &Element = resource_reference.downcast_ref().unwrap();
        let node: &'static Node =
            unsafe { mem::transmute::<&Node, &'static Node>(element.as_ref()) };

        Ok(node)
    } else if resource_type_id == TypeId::of::<HtmlElement>() {
        let html_element: &HtmlElement = resource_reference.downcast_ref().unwrap();
        let node: &'static Node =
            unsafe { mem::transmute::<&Node, &'static Node>(html_element.as_ref()) };

        Ok(node)
    } else if resource_type_id == TypeId::of::<Text>() {
        let text: &Text = resource_reference.downcast_ref().unwrap();
        let node: &'static Node = unsafe { mem::transmute::<&Node, &'static Node>(text.as_ref()) };

        Ok(node)
    } else {
        Err(badarg!().into())
    }
}
