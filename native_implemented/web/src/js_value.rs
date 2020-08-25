use std::str;

use wasm_bindgen::JsValue;

use web_sys::{
    Document, Element, HtmlBodyElement, HtmlElement, HtmlTableElement, Node, Text, WebSocket,
};

use liblumen_alloc::erts::term::prelude::*;

pub fn from_term(term: Term) -> JsValue {
    match term.decode().unwrap() {
        TypedTerm::Atom(atom) => from_atom(atom),
        TypedTerm::HeapBinary(heap_binary) => from_aligned_binary(heap_binary),
        TypedTerm::ProcBin(process_binary) => from_aligned_binary(process_binary),
        TypedTerm::ResourceReference(resource_reference) => {
            from_resource_reference(resource_reference.into())
        }
        TypedTerm::Tuple(tuple) => from_tuple(&tuple),
        TypedTerm::Pid(pid) => from_pid(pid),
        TypedTerm::SmallInteger(small_integer) => from_small_integer(small_integer),
        _ => unimplemented!("Convert {:?} to JsValue", term),
    }
}

// Private

fn from_aligned_binary<A: AlignedBinary>(aligned_binary: A) -> JsValue {
    from_bytes(aligned_binary.as_bytes())
}

fn from_atom(atom: Atom) -> JsValue {
    js_sys::Symbol::for_(atom.name()).into()
}

fn from_bytes(bytes: &[u8]) -> JsValue {
    match str::from_utf8(bytes) {
        Ok(s) => s.into(),
        Err(_) => {
            let uint8_array = unsafe { js_sys::Uint8Array::view(bytes) };

            uint8_array.into()
        }
    }
}

fn from_pid(pid: Pid) -> JsValue {
    let array = js_sys::Array::new();

    array.push(&(pid.number() as i32).into());
    array.push(&(pid.serial() as i32).into());

    array.into()
}

fn from_resource_reference(resource_reference: Resource) -> JsValue {
    if resource_reference.is::<Document>() {
        let document: &Document = resource_reference.downcast_ref().unwrap();

        document.into()
    } else if resource_reference.is::<Element>() {
        let element: &Element = resource_reference.downcast_ref().unwrap();

        element.into()
    } else if resource_reference.is::<HtmlBodyElement>() {
        let html_body_element: &HtmlBodyElement = resource_reference.downcast_ref().unwrap();

        html_body_element.into()
    } else if resource_reference.is::<HtmlElement>() {
        let html_element: &HtmlElement = resource_reference.downcast_ref().unwrap();

        html_element.into()
    } else if resource_reference.is::<HtmlTableElement>() {
        let html_table_element: &HtmlTableElement = resource_reference.downcast_ref().unwrap();

        html_table_element.into()
    } else if resource_reference.is::<Node>() {
        let node: &Node = resource_reference.downcast_ref().unwrap();

        node.into()
    } else if resource_reference.is::<Text>() {
        let text: &Text = resource_reference.downcast_ref().unwrap();

        text.into()
    } else if resource_reference.is::<WebSocket>() {
        let web_socket: &WebSocket = resource_reference.downcast_ref().unwrap();

        web_socket.into()
    } else {
        unimplemented!("Convert {:?} to JsValue", resource_reference);
    }
}

fn from_small_integer(small_integer: SmallInteger) -> JsValue {
    let i: isize = small_integer.into();

    if (std::i32::MIN as isize) <= i && i <= (std::i32::MAX as isize) {
        (i as i32).into()
    } else {
        (i as f64).into()
    }
}

fn from_tuple(tuple: &Tuple) -> JsValue {
    let array = js_sys::Array::new();

    for element_term in tuple.iter() {
        let element_js_value = from_term(*element_term);
        array.push(&element_js_value);
    }

    array.into()
}
