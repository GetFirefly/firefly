#[path = "class_name_1/label_1.rs"]
pub mod label_1;
#[path = "class_name_1/label_2.rs"]
pub mod label_2;
#[path = "class_name_1/label_3.rs"]
pub mod label_3;

use super::*;

use liblumen_alloc::erts::term::prelude::Atom;
use liblumen_alloc::erts::process::{Frame, Native};
use liblumen_alloc::erts::{Arity, ModuleFunctionArity};

use liblumen_web::window;

#[wasm_bindgen_test(async)]
fn with_class_name_returns_class_name() -> impl Future<Item = (), Error = JsValue> {
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

    JsFuture::from(promise)
        .map(move |resolved| {
            let class_name_js_string: JsValue = class_name.into();

            assert_eq!(resolved, class_name_js_string);
        })
        .map_err(|_| unreachable!())
}

#[wasm_bindgen_test(async)]
fn with_class_names_returns_space_separateed_class_names() -> impl Future<Item = (), Error = JsValue>
{
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

    JsFuture::from(promise)
        .map(move |resolved| {
            let class_name_js_string: JsValue = class_name.into();

            assert_eq!(resolved, class_name_js_string);
        })
        .map_err(|_| unreachable!())
}

#[wasm_bindgen_test(async)]
fn without_class_returns_empty_list() -> impl Future<Item = (), Error = JsValue> {
    start_once();

    let promise = promise();

    JsFuture::from(promise)
        .map(move |resolved| {
            let empty_js_string: JsValue = "".into();

            assert_eq!(resolved, empty_js_string);
        })
        .map_err(|_| unreachable!())
}

// Private

const ARITY: Arity = 0;

fn frame(native: Native) -> Frame {
    Frame::new(module_function_arity(), native)
}

fn function() -> Atom {
    Atom::from_str("class_name_1")
}

fn module() -> Atom {
    Atom::from_str("Lumen.Web.ElementTest")
}

fn module_function_arity() -> ModuleFunctionArity {
    ModuleFunctionArity {
        module: module(),
        function: function(),
        arity: ARITY,
    }
}

fn promise() -> js_sys::Promise {
    let options: Options = Default::default();

    // ```elixir
    // {:ok, window} = Lumen.Web.Window.window()
    // {:ok, document} = Lumen.Web.Window.document(window)
    // {:ok, body} = Lumen.Web.Document.body(document)
    // class_name = Lumen.Web.Element.class_name(body)
    // Lumen.Web.Wait.with_return(class_name)
    // ```
    wait::with_return_0::spawn(options, |_| {
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
            // {:ok, body} = Lumen.Web.Document.body(document)
            // class_name = Lumen.Web.Element.class_name(body)
            // Lumen.Web.Wait.with_return(class_name)
            // ```
            label_1::frame().with_arguments(true, &[]),
        ])
    })
    .unwrap()
}
