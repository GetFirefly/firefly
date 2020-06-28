#[path = "remove_1/removes_element.rs"]
mod removes_element;

use super::*;

use js_sys::Symbol;

use liblumen_web::window;

#[wasm_bindgen_test(async)]
fn removes_element() -> impl Future<Item = (), Error = JsValue> {
    start_once();

    let options: Options = Default::default();

    // ```elixir
    // {:ok, window} = Lumen.Web.Window.window()
    // {:ok, document} = Lumen.Web.Window.document(window)
    // {:ok, body} = Lumen.Web.Document.body(document)
    // {:ok, child} = Lumen.Web.Document.create_element(body, "table");
    // :ok = Lumen.Web.Node.append_child(document, child);
    // :ok = Lumen.Web.Element.remove(child);
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
            // {:ok, body} = Lumen.Web.Document.body(document)
            // {:ok, child} = Lumen.Web.Document.create_element(body, "table");
            // :ok = Lumen.Web.Node.append_child(body, child);
            // :ok = Lumen.Web.Element.remove(child);
            // Lumen.Web.Wait.with_return(body_tuple)
            // ```
            removes_element::label_1::frame().with_arguments(true, &[]),
        ])
    })
    .unwrap();

    JsFuture::from(promise)
        .map(move |resolved| {
            let ok: JsValue = Symbol::for_("ok").into();

            assert_eq!(resolved, ok);
        })
        .map_err(|_| unreachable!())
}
