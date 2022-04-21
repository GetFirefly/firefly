#[path = "remove_1/removes_element.rs"]
pub mod removes_element;

use super::*;

use js_sys::Symbol;

#[wasm_bindgen_test]
async fn removes_element() {
    start_once();

    let promise = r#async::apply_3::promise(
        module(),
        removes_element::function(),
        vec![],
        Default::default(),
    )
    .unwrap();
    let resolved = JsFuture::from(promise).await.unwrap();

    let ok: JsValue = Symbol::for_("ok").into();

    assert_eq!(resolved, ok);
}

fn module() -> Atom {
    Atom::from_str("Elixir.Lumen.Web.Element.Remove1")
}
