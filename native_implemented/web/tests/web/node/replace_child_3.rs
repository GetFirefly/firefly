#[path = "replace_child_3/with_new_child_is_parent_returns_error_hierarchy_request.rs"]
pub mod with_new_child_is_parent_returns_error_hierarchy_request;
#[path = "replace_child_3/with_new_child_returns_ok_replaced_child.rs"]
pub mod with_new_child_returns_ok_replaced_child;

use super::*;

use wasm_bindgen::JsCast;

use js_sys::{Reflect, Symbol};

use web_sys::Element;

#[wasm_bindgen_test(async)]
fn with_new_child_is_parent_returns_error_hierarchy_request(
) -> impl Future<Item = (), Error = JsValue> {
    start_once();

    let promise = r#async::apply_3::promise(
        module(),
        with_new_child_is_parent_returns_error_hierarchy_request::function(),
        vec![],
        Default::default(),
    )
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

    let promise = r#async::apply_3::promise(
        module(),
        with_new_child_returns_ok_replaced_child::function(),
        vec![],
        Default::default(),
    )
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

fn module() -> Atom {
    Atom::from_str("Elixir.Lumen.Web.Node.ReplaceChild3")
}

fn module_id() -> usize {
    module().id()
}
