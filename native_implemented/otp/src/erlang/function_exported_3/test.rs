use liblumen_alloc::atom;

use crate::erlang::function_exported_3::result;

#[test]
fn without_exported_function_returns_false() {
    let module = atom!("unexported_module");
    let function = atom!("unexported_function");
    let arity = 0.into();

    assert_eq!(result(module, function, arity), Ok(false.into()));
}

#[test]
fn with_exported_function_return_true() {
    let module = atom!("erlang");
    let function = atom!("self");
    let arity = 0.into();

    assert_eq!(result(module, function, arity), Ok(true.into()));
}
