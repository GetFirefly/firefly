use liblumen_alloc::atom;

use crate::erlang::function_exported_3::native;

#[test]
fn without_exported_function_returns_false() {
    let module = atom!("unexported_module");
    let function = atom!("unexported_function");
    let arity = 0.into();

    assert_eq!(native(module, function, arity), Ok(false.into()));
}

#[test]
fn with_exported_function_return_true() {
    let module = atom!("erlang");
    let function = atom!("self");
    let arity = 0.into();

    crate::erlang::self_0::export();

    assert_eq!(native(module, function, arity), Ok(true.into()));
}
