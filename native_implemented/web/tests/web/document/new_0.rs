use super::*;

use js_sys::{Reflect, Symbol};

use wasm_bindgen::JsCast;

use web_sys::Document;

use liblumen_web::document;

#[wasm_bindgen_test(async)]
fn returns_ok_tuple() -> impl Future<Item = (), Error = JsValue> {
    start_once();

    // ```elixir
    // document_tuple = Lumen.Web.Document.new()
    // Lumen.Web.Wait.with_return(document_tuple)
    // ```
    let promise = r#async::apply_3::promise(
        document::module(),
        document::new_0::function(),
        vec![],
        Default::default(),
    )
    .unwrap();

    JsFuture::from(promise)
        .map(move |resolved| {
            assert!(js_sys::Array::is_array(&resolved));

            let resolved_array: js_sys::Array = resolved.dyn_into().unwrap();

            assert_eq!(resolved_array.length(), 2);

            let ok: JsValue = Symbol::for_("ok").into();
            assert_eq!(Reflect::get(&resolved_array, &0.into()).unwrap(), ok);

            assert!(Reflect::get(&resolved_array, &1.into())
                .unwrap()
                .has_type::<Document>());
        })
        .map_err(|_| unreachable!())
}
