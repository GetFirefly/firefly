#[path = "replace_child_3/with_new_child_is_parent_returns_error_hierarchy_request.rs"]
mod with_new_child_is_parent_returns_error_hierarchy_request;
#[path = "replace_child_3/with_new_child_returns_ok_replaced_child.rs"]
mod with_new_child_returns_ok_replaced_child;

use super::*;

use wasm_bindgen::JsCast;

use js_sys::{Reflect, Symbol};

use web_sys::Element;

use liblumen_web::document;

#[wasm_bindgen_test(async)]
fn with_new_child_is_parent_returns_error_hierarchy_request(
) -> impl Future<Item = (), Error = JsValue> {
    start_once();

    let options: Options = Default::default();

    // ```elixir
    // {:ok, document} = Lumen.Web.Document.new()
    // {:ok, old_child} = Lumen.Web.Document.create_element(document, "table")
    // {:ok, parent} = Lumen.Web.Document.create_element(document, "div")
    // :ok = Lumen.Web.Node.append_child(parent, old_child)
    // {:error, :hierarchy_request} = Lumen.Web.replace_child(parent, old_child, parent)
    // ```
    let promise = wait::with_return_0::spawn(options, |_| {
        Ok(vec![
            // ```elixir
            // # pushed to stack: ()
            // # returned from call: N/A
            // # full stack: ()
            // # returns: {:ok, document}
            // ```
            document::new_0::frame().with_arguments(false, &[]),
            // ```elixir
            // # label 1
            // # pushed to stack: ()
            // # returned form call: {:ok, document}
            // # full stack: ({:ok, document})
            // # returns: {:ok parent}
            // {:ok, old_child} = Lumen.Web.Document.create_element(document, "table")
            // {:ok, parent} = Lumen.Web.Document.create_element(document, "div")
            // :ok = Lumen.Web.Node.append_child(parent, old_child)
            // {:error, :hierarchy_request} = Lumen.Web.replace_child(parent, old_child, parent)
            // ```
            with_new_child_is_parent_returns_error_hierarchy_request::label_1::frame()
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

            let error: JsValue = Symbol::for_("error").into();
            assert_eq!(Reflect::get(&resolved_array, &0.into()).unwrap(), error);

            let hierarchy_request: JsValue = Symbol::for_("hierarchy_request").into();
            assert_eq!(
                Reflect::get(&resolved_array, &1.into()).unwrap(),
                hierarchy_request
            );
        })
        .map_err(|_| unreachable!())
}

#[wasm_bindgen_test(async)]
fn with_new_child_returns_ok_replaced_child() -> impl Future<Item = (), Error = JsValue> {
    start_once();

    let options: Options = Default::default();

    // ```elixir
    // {:ok, document} = Lumen.Web.Document.new()
    // {:ok, old_child} = Lumen.Web.Document.create_element(document, "table")
    // {:ok, parent} = Lumen.Web.Document.create_element(parent_document, "div")
    // :ok = Lumen.Web.Node.append_child(parent, old_child)
    // {:ok, new_child} = Lumen.Web.Document.create_element(document, "ul");
    // {:ok, replaced_child} = Lumen.Web.replace_child(parent, new_child, old_child)
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
            // # returns: {:ok parent}
            // {:ok, old_child} = Lumen.Web.Document.create_element(document, "table")
            // {:ok, parent} = Lumen.Web.Document.create_element(parent_document, "div")
            // :ok = Lumen.Web.Node.append_child(parent, old_child)
            // {:ok, new_child} = Lumen.Web.Document.create_element(document, "ul");
            // {:ok, replaced_child} = Lumen.Web.replace_child(parent, new_child, old_child)
            // ```
            with_new_child_returns_ok_replaced_child::label_1::frame().with_arguments(true, &[]),
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

            let replaced_child = Reflect::get(&resolved_array, &1.into()).unwrap();

            assert!(replaced_child.has_type::<Element>());

            let replaced_element: Element = replaced_child.dyn_into().unwrap();

            assert_eq!(replaced_element.tag_name(), "table");
        })
        .map_err(|_| unreachable!())
}
