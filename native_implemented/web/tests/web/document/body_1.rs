#[path = "body_1/without_body.rs"]
pub mod without_body;
#[path = "body_1/with_body.rs"]
pub mod with_body;

use super::*;

use js_sys::{Reflect, Symbol};

use wasm_bindgen::JsCast;

use liblumen_alloc::erts::term::prelude::*;

#[wasm_bindgen_test(async)]
fn without_body() -> impl Future<Item = (), Error = JsValue> {
    start_once();

    let promise = r#async::apply_3::promise(
        module(),
        without_body::function(),
        vec![],
        Default::default(),
    )
    .unwrap();

    JsFuture::from(promise)
        .map(move |resolved| {
            let error: JsValue = Symbol::for_("error").into();

            assert_eq!(resolved, error);
        })
        .map_err(|_| unreachable!())
}

#[wasm_bindgen_test(async)]
fn with_body() -> impl Future<Item = (), Error = JsValue> {
    start_once();

    let promise =
        r#async::apply_3::promise(module(), with_body::function(), vec![], Default::default())
            .unwrap();

    JsFuture::from(promise)
        .map(move |resolved| {
            assert!(js_sys::Array::is_array(&resolved));

            let resolved_array: js_sys::Array = resolved.dyn_into().unwrap();

            assert_eq!(resolved_array.length(), 2);

            let ok: JsValue = Symbol::for_("ok").into();
            assert_eq!(Reflect::get(&resolved_array, &0.into()).unwrap(), ok);

            let body: JsValue = web_sys::window()
                .unwrap()
                .document()
                .unwrap()
                .body()
                .unwrap()
                .into();
            assert_eq!(Reflect::get(&resolved_array, &1.into()).unwrap(), body);
        })
        .map_err(|_| unreachable!())
}

fn module() -> Atom {
    Atom::from_str("Elixir.Lumen.Web.Document.Body1")
}

fn module_id() -> usize {
    module().id()
}
