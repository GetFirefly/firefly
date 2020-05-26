#[path = "body_1/without_body.rs"]
mod without_body;

#[path = "body_1/with_body.rs"]
mod with_body;

use super::*;

use js_sys::{Reflect, Symbol};

use wasm_bindgen::JsCast;


use liblumen_alloc::erts::process::{Frame, Native};
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::erts::{ModuleFunctionArity, Arity};

use liblumen_web::{document, window};

#[wasm_bindgen_test(async)]
fn without_body() -> impl Future<Item = (), Error = JsValue> {
    start_once();

    let options: Options = Default::default();

    // ```elixir
    // {:ok, document} = Lumen.Web.Document.new()
    // body_tuple = Lumen.Web.Document.body(document)
    // Lumen.Web.Wait.with_return(body_tuple)
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
            // # returned from call: {:ok, document}
            // # full stack: ({:ok, document})
            // # returns: {:ok, body} | :error
            // body_tuple = Lumen.Web.Document.body(document)
            // Lumen.Web.Wait.with_return(body_tuple)
            // ```
            without_body::label_1::frame().with_arguments(true, &[]),
        ])
    })
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

    let options: Options = Default::default();

    // ```elixir
    // {:ok, document} = Lumen.Web.Document.new()
    // body_tuple = Lumen.Web.Document.body(document)
    // Lumen.Web.Wait.with_return(body_tuple)
    // ```
    let promise = wait::with_return_0::spawn(options, |_| {
        Ok(vec![
            // ```elixir
            // # pushed to stack: ()
            // # returned from call: N/A
            // # full stack: ()
            // # returns: {:ok, window}
            // ```
            window::window_0::frame().with_arguments(false, &[]),
            // ```elixir
            // # label 1
            // # pushed to stack: ()
            // # returned from call: {:ok, window}
            // # full stack: ({:ok, window})
            // # returns: {:ok, document}
            // {:ok, document} = Lumen.Web.Window.document(window)
            // body_tuple = Lumen.Web.Document.body(document)
            // Lumen.Web.Wait.with_return(body_tuple)
            // ```
            with_body::label_1::frame().with_arguments(true, &[]),
        ])
    })
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

const ARITY: Arity = 1;
