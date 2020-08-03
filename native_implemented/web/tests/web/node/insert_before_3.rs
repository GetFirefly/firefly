#[path = "insert_before_3/with_nil_reference_child_appends_new_child.rs"]
mod with_nil_reference_child_appends_new_child;
#[path = "insert_before_3/with_reference_child_inserts_before_reference_child.rs"]
mod with_reference_child_inserts_before_reference_child;

use super::*;

use wasm_bindgen::JsCast;

use js_sys::{Reflect, Symbol};

use web_sys::Element;

use liblumen_web::document;

#[wasm_bindgen_test(async)]
fn with_nil_reference_child_appends_new_child() -> impl Future<Item = (), Error = JsValue> {
    start_once();

    let options: Options = Default::default();

    // ```elixir
    // {:ok, document} = Lumen.Web.Document.new()
    // {:ok, existing_child} = Lumen.Web.Document.create_element(document, "table")
    // {:ok, parent} = Lumen.Web.Document.create_element(parent_document, "div")
    // :ok = Lumen.Web.Node.append_child(document, parent)
    // :ok = Lumen.Web.Node.append_child(parent, existing_child)
    // {:ok, new_child} = Lumen.Web.Document.create_element(document, "ul");
    // {:ok, inserted_child} = Lumen.Web.insert_before(parent, new_child, nil)
    // ```
    let promise = wait::with_return_0::spawn(options, |_| {
        Ok(vec![
            // ```elixir
            // # pushed to stack: ()
            // # returned from call: N/A
            // # full stack: ()
            // # returns: {:ok, parent_document}
            // ```
            document::new_0::frame().with_arguments(false, &[]),
            // ```elixir
            // # label 1
            // # pushed to stack: ()
            // # returned form call: {:ok, document}
            // # full stack: ({:ok, document})
            // # returns: {:ok, old_child}
            // {:ok, existing_child} = Lumen.Web.Document.create_element(document, "table")
            // {:ok, parent} = Lumen.Web.Document.create_element(parent_document, "div")
            // :ok = Lumen.Web.Node.append_child(document, parent)
            // :ok = Lumen.Web.Node.append_child(parent, existing_child)
            // {:ok, new_child} = Lumen.Web.Document.create_element(document, "ul");
            // {:ok, inserted_child} = Lumen.Web.insert_before(parent, new_child, nil)
            // ```
            with_nil_reference_child_appends_new_child::label_1::frame().with_arguments(true, &[]),
        ])
    })
    .unwrap();

    JsFuture::from(promise)
        .map(move |resolved| {
            assert!(
                js_sys::Array::is_array(&resolved),
                "{:?} is not an array",
                resolved
            );

            let resolved_array: js_sys::Array = resolved.dyn_into().unwrap();

            assert_eq!(resolved_array.length(), 2);

            let ok: JsValue = Symbol::for_("ok").into();
            assert_eq!(Reflect::get(&resolved_array, &0.into()).unwrap(), ok);

            let inserted_child = Reflect::get(&resolved_array, &1.into()).unwrap();

            assert!(inserted_child.has_type::<Element>());

            let inserted_element: Element = inserted_child.dyn_into().unwrap();

            assert_eq!(inserted_element.tag_name(), "ul");

            let previous_element_sibling = inserted_element.previous_element_sibling().unwrap();

            assert_eq!(previous_element_sibling.tag_name(), "table");
        })
        .map_err(|_| unreachable!())
}

#[wasm_bindgen_test(async)]
fn with_reference_child_inserts_before_reference_child() -> impl Future<Item = (), Error = JsValue>
{
    start_once();

    let options: Options = Default::default();

    // ```elixir
    // {:ok, document} = Lumen.Web.Document.new()
    // {:ok, reference_child} = Lumen.Web.Document.create_element(document, "table")
    // {:ok, parent} = Lumen.Web.Document.create_element(parent_document, "div")
    // :ok = Lumen.Web.Node.append_child(document, parent)
    // :ok = Lumen.Web.Node.append_child(parent, reference_child)
    // {:ok, new_child} = Lumen.Web.Document.create_element(document, "ul");
    // {:ok, inserted_child} = Lumen.Web.insert_before(parent, new_child, reference_child)
    // ```
    let promise = wait::with_return_0::spawn(options, |_| {
        Ok(vec![
            // ```elixir
            // # pushed to stack: ()
            // # returned from call: N/A
            // # full stack: ()
            // # returns: {:ok, parent_document}
            // ```
            document::new_0::frame().with_arguments(false, &[]),
            // ```elixir
            // # label 1
            // # pushed to stack: ()
            // # returned form call: {:ok, document}
            // # full stack: ({:ok, document})
            // # returns: {:ok, old_child}
            // {:ok, reference_child} = Lumen.Web.Document.create_element(document, "table")
            // {:ok, parent} = Lumen.Web.Document.create_element(parent_document, "div")
            // :ok = Lumen.Web.Node.append_child(document, parent)
            // :ok = Lumen.Web.Node.append_child(parent, reference_child)
            // {:ok, new_child} = Lumen.Web.Document.create_element(document, "ul");
            // {:ok, inserted_child} = Lumen.Web.insert_before(parent, new_child, reference_child)
            // ```
            with_reference_child_inserts_before_reference_child::label_1::frame()
                .with_arguments(true, &[]),
        ])
    })
    .unwrap();

    JsFuture::from(promise)
        .map(move |resolved| {
            assert!(
                js_sys::Array::is_array(&resolved),
                "{:?} is not an array",
                resolved
            );

            let resolved_array: js_sys::Array = resolved.dyn_into().unwrap();

            assert_eq!(resolved_array.length(), 2);

            let ok: JsValue = Symbol::for_("ok").into();
            assert_eq!(Reflect::get(&resolved_array, &0.into()).unwrap(), ok);

            let inserted_child = Reflect::get(&resolved_array, &1.into()).unwrap();

            assert!(inserted_child.has_type::<Element>());

            let inserted_element: Element = inserted_child.dyn_into().unwrap();

            assert_eq!(inserted_element.tag_name(), "ul");

            let next_element_sibling = inserted_element.next_element_sibling().unwrap();

            assert_eq!(next_element_sibling.tag_name(), "table");
        })
        .map_err(|_| unreachable!())
}
