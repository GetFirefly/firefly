//! Test suite for the Web and headless browsers.
#![cfg(target_arch = "wasm32")]

extern crate wasm_bindgen_test;

use std::sync::Once;

use futures::future::Future;

use wasm_bindgen::JsValue;

use wasm_bindgen_futures::JsFuture;

use wasm_bindgen_test::*;

use spawn_chain::start;

wasm_bindgen_test_configure!(run_in_browser);

mod run {
    use super::*;

    use spawn_chain::run;

    #[wasm_bindgen_test(async)]
    fn with_1() -> impl Future<Item = (), Error = JsValue> {
        eq_in_the_future(1)
    }

    #[wasm_bindgen_test(async)]
    fn with_2() -> impl Future<Item = (), Error = JsValue> {
        eq_in_the_future(2)
    }

    #[wasm_bindgen_test(async)]
    fn with_4() -> impl Future<Item = (), Error = JsValue> {
        eq_in_the_future(4)
    }

    #[wasm_bindgen_test(async)]
    fn with_8() -> impl Future<Item = (), Error = JsValue> {
        eq_in_the_future(8)
    }

    #[wasm_bindgen_test(async)]
    fn with_16() -> impl Future<Item = (), Error = JsValue> {
        eq_in_the_future(16)
    }

    #[wasm_bindgen_test(async)]
    fn with_32() -> impl Future<Item = (), Error = JsValue> {
        eq_in_the_future(32)
    }

    fn eq_in_the_future(n: usize) -> impl Future<Item = (), Error = JsValue> {
        super::eq_in_the_future(run, n)
    }
}

mod log_to_console {
    use super::*;

    use spawn_chain::log_to_console;

    #[wasm_bindgen_test(async)]
    fn with_1() -> impl Future<Item = (), Error = JsValue> {
        eq_in_the_future(1)
    }

    #[wasm_bindgen_test(async)]
    fn with_2() -> impl Future<Item = (), Error = JsValue> {
        eq_in_the_future(2)
    }

    #[wasm_bindgen_test(async)]
    fn with_4() -> impl Future<Item = (), Error = JsValue> {
        eq_in_the_future(4)
    }

    #[wasm_bindgen_test(async)]
    fn with_8() -> impl Future<Item = (), Error = JsValue> {
        eq_in_the_future(8)
    }

    #[wasm_bindgen_test(async)]
    fn with_16() -> impl Future<Item = (), Error = JsValue> {
        eq_in_the_future(16)
    }

    #[wasm_bindgen_test(async)]
    fn with_32() -> impl Future<Item = (), Error = JsValue> {
        eq_in_the_future(32)
    }

    fn eq_in_the_future(n: usize) -> impl Future<Item = (), Error = JsValue> {
        super::eq_in_the_future(log_to_console, n)
    }
}

mod log_to_dom {
    use super::*;

    use spawn_chain::log_to_dom;

    #[wasm_bindgen_test(async)]
    fn with_1() -> impl Future<Item = (), Error = JsValue> {
        eq_in_the_future(1)
    }

    #[wasm_bindgen_test(async)]
    fn with_2() -> impl Future<Item = (), Error = JsValue> {
        eq_in_the_future(2)
    }

    #[wasm_bindgen_test(async)]
    fn with_4() -> impl Future<Item = (), Error = JsValue> {
        eq_in_the_future(4)
    }

    #[wasm_bindgen_test(async)]
    fn with_8() -> impl Future<Item = (), Error = JsValue> {
        eq_in_the_future(8)
    }

    #[wasm_bindgen_test(async)]
    fn with_16() -> impl Future<Item = (), Error = JsValue> {
        eq_in_the_future(16)
    }

    #[wasm_bindgen_test(async)]
    fn with_32() -> impl Future<Item = (), Error = JsValue> {
        eq_in_the_future(32)
    }

    fn eq_in_the_future(n: usize) -> impl Future<Item = (), Error = JsValue> {
        super::eq_in_the_future(log_to_dom, n)
    }
}

static START: Once = Once::new();

fn eq_in_the_future(
    f: fn(usize) -> js_sys::Promise,
    n: usize,
) -> impl Future<Item = (), Error = JsValue> {
    start_once();

    let promise = f(n);

    JsFuture::from(promise)
        .map(move |resolved| {
            let n_js_value: JsValue = (n as i32).into();

            assert_eq!(resolved, n_js_value)
        })
        .map_err(|_| unreachable!())
}

fn start_once() {
    START.call_once(|| {
        start();
    })
}
