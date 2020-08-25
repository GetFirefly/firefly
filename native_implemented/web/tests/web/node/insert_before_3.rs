#[path = "insert_before_3/with_nil_reference_child_appends_new_child.rs"]
pub mod with_nil_reference_child_appends_new_child;
#[path = "insert_before_3/with_reference_child_inserts_before_reference_child.rs"]
pub mod with_reference_child_inserts_before_reference_child;

use super::*;

use wasm_bindgen::JsCast;

use js_sys::{Reflect, Symbol};

use web_sys::Element;

#[wasm_bindgen_test]
async fn with_nil_reference_child_appends_new_child() {
    start_once();

    let promise = r#async::apply_3::promise(
        module(),
        with_nil_reference_child_appends_new_child::function(),
        vec![],
        Default::default(),
    )
    .unwrap();
    let resolved = JsFuture::from(promise).await.unwrap();

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
}

#[wasm_bindgen_test]
async fn with_reference_child_inserts_before_reference_child() {
    start_once();

    let promise = r#async::apply_3::promise(
        module(),
        with_reference_child_inserts_before_reference_child::function(),
        vec![],
        Default::default(),
    )
    .unwrap();
    let resolved = JsFuture::from(promise).await.unwrap();

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
}

fn module() -> Atom {
    Atom::from_str("Elixir.Lumen.Web.Node.InsertBefore3")
}

fn module_id() -> usize {
    module().id()
}
