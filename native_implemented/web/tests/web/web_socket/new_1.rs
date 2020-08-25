use super::*;

use js_sys::{Reflect, Symbol};

use wasm_bindgen::JsCast;

use web_sys::WebSocket;

use liblumen_alloc::erts::fragment::HeapFragment;

use liblumen_web::web_socket;

#[wasm_bindgen_test(async)]
fn with_valid_url_returns_ok_tuple() -> impl Future<Item = (), Error = JsValue> {
    start_once();

    let (url, url_non_null_heap_fragment) =
        HeapFragment::new_binary_from_str("wss://echo.websocket.org").unwrap();
    let options: Options = Default::default();

    // ```elixir
    // web_socket_tuple = Lumen.Web.WebSocket.new(url)
    // Lumen.Web.Wait.with_return(web_socket_tuple)
    // ```
    let promise = r#async::apply_3::promise(
        web_socket::module(),
        web_socket::new_1::function(),
        vec![url],
        options,
    )
    .unwrap();

    std::mem::drop(url_non_null_heap_fragment);

    JsFuture::from(promise)
        .map(move |resolved| {
            assert!(js_sys::Array::is_array(&resolved));

            let resolved_array: js_sys::Array = resolved.dyn_into().unwrap();

            assert_eq!(resolved_array.length(), 2);

            let ok: JsValue = Symbol::for_("ok").into();
            assert_eq!(Reflect::get(&resolved_array, &0.into()).unwrap(), ok);

            assert!(Reflect::get(&resolved_array, &1.into())
                .unwrap()
                .has_type::<WebSocket>());
        })
        .map_err(|_| unreachable!())
}

#[wasm_bindgen_test(async)]
fn without_valid_url_returns_error_tuple() -> impl Future<Item = (), Error = JsValue> {
    start_once();

    let (url, url_non_null_heap_fragment) =
        HeapFragment::new_binary_from_str("invalid_url").unwrap();
    let options: Options = Default::default();

    // ```elixir
    // web_socket_tuple = Lumen.Web.WebSocket.new(url)
    // Lumen.Web.Wait.with_return(web_socket_tuple)
    // ```
    let promise = r#async::apply_3::promise(
        web_socket::module(),
        web_socket::new_1::function(),
        vec![url],
        options,
    )
    .unwrap();

    std::mem::drop(url_non_null_heap_fragment);

    JsFuture::from(promise)
        .map(move |resolved| {
            assert!(js_sys::Array::is_array(&resolved));

            let resolved_array: js_sys::Array = resolved.dyn_into().unwrap();

            assert_eq!(resolved_array.length(), 2);

            let error: JsValue = Symbol::for_("error").into();
            assert_eq!(Reflect::get(&resolved_array, &0.into()).unwrap(), error);

            let reason = Reflect::get(&resolved_array, &1.into()).unwrap();

            let reason_array: js_sys::Array = reason.dyn_into().unwrap();

            assert_eq!(reason_array.length(), 2);

            let tag: JsValue = Symbol::for_("syntax").into();
            assert_eq!(Reflect::get(&reason_array, &0.into()).unwrap(), tag);
            assert!(Reflect::get(&reason_array, &1.into()).unwrap().is_string());
        })
        .map_err(|_| unreachable!())
}
