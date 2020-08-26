#[path = "random_integer_1/returns_integer_between_0_inclusive_and_max_exclusive.rs"]
pub mod returns_integer_between_0_inclusive_and_max_exclusive;

use self::returns_integer_between_0_inclusive_and_max_exclusive::EXCLUSIVE_MAX;
use super::*;

#[wasm_bindgen_test]
async fn returns_integer_between_0_inclusive_and_max_exclusive() {
    start_once();

    let promise = r#async::apply_3::promise(
        module(),
        returns_integer_between_0_inclusive_and_max_exclusive::function(),
        vec![],
        Default::default(),
    )
    .unwrap();
    let resolved = JsFuture::from(promise).await.unwrap();

    assert!(
        js_sys::Number::is_integer(&resolved),
        "{:?} is not an integer",
        resolved
    );

    let resolved_usize = resolved.as_f64().unwrap() as usize;

    assert!(resolved_usize < EXCLUSIVE_MAX);
}

fn module() -> Atom {
    Atom::from_str("Elixir.Lumen.Web.Math.RandomInteger1")
}

fn module_id() -> usize {
    module().id()
}
