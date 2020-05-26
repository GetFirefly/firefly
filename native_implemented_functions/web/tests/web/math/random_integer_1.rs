use super::*;

#[wasm_bindgen_test(async)]
fn returns_integer_between_0_inclusive_and_max_exclusive() -> impl Future<Item = (), Error = JsValue>
{
    start_once();

    let options: Options = Default::default();

    let exclusive_max = 2;

    // ```elixir
    // {:ok, document} = Lumen.Web.Document.new()
    // {:ok, old_child} = Lumen.Web.Document.create_element(document, "table")
    // {:ok, parent} = Lumen.Web.Document.create_element(document, "div")
    // :ok = Lumen.Web.Node.append_child(parent, old_child)
    // {:error, :hierarchy_request} = Lumen.Web.replace_child(parent, old_child, parent)
    // ```
    let promise = wait::with_return_0::spawn(options, |child_process| {
        Ok(vec![
            // ```elixir
            // # pushed to stack: ()
            // # returned from call: N/A
            // # full stack: ()
            // # returns: {:ok, document}
            // ```
            liblumen_web::math::random_integer_1::frame()
                .with_arguments(false, &[child_process.integer(exclusive_max).unwrap()]),
        ])
    })
    .unwrap();

    JsFuture::from(promise)
        .map(move |resolved| {
            assert!(
                js_sys::Number::is_integer(&resolved),
                "{:?} is not an integer",
                resolved
            );

            let resolved_usize = resolved.as_f64().unwrap() as usize;

            assert!(resolved_usize < exclusive_max);
        })
        .map_err(|_| unreachable!())
}
