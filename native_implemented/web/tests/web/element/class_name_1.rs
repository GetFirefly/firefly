#[path = "class_name_1/test_0.rs"]
pub mod test_0;

use super::*;

use liblumen_alloc::erts::term::prelude::Atom;

#[wasm_bindgen_test]
async fn with_class_name_returns_class_name() {
    start_once();

    let body = web_sys::window()
        .unwrap()
        .document()
        .unwrap()
        .body()
        .unwrap();
    let class_name = "is-ready";
    body.set_class_name(class_name);

    let promise = promise();
    let resolved = JsFuture::from(promise).await.unwrap();

    let class_name_js_string: JsValue = class_name.into();

    assert_eq!(resolved, class_name_js_string);
}

#[wasm_bindgen_test]
async fn with_class_names_returns_space_separateed_class_names() {
    start_once();

    let body = web_sys::window()
        .unwrap()
        .document()
        .unwrap()
        .body()
        .unwrap();
    let class_name = "is-ready classy";
    body.set_class_name(class_name);

    let promise = promise();
    let resolved = JsFuture::from(promise).await.unwrap();

    let class_name_js_string: JsValue = class_name.into();

    assert_eq!(resolved, class_name_js_string);
}

#[wasm_bindgen_test]
async fn without_class_returns_empty_list() {
    start_once();

    let promise = promise();
    let resolved = JsFuture::from(promise).await.unwrap();

    let empty_js_string: JsValue = "".into();

    assert_eq!(resolved, empty_js_string);
}

fn module() -> Atom {
    Atom::from_str("Elixir.Lumen.Web.Element.ClassName1")
}

fn promise() -> js_sys::Promise {
    r#async::apply_3::promise(module(), test_0::function(), vec![], Default::default()).unwrap()
}
