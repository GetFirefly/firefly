use liblumen_alloc::erts::term::atom_unchecked;

use crate::otp::erlang::function_exported_3::native;

#[test]
fn without_exported_function_returns_false() {
    let module = atom_unchecked("unexported_module");
    let function = atom_unchecked("unexported_function");
    let arity = 0.into();

    assert_eq!(native(module, function, arity), Ok(false.into()));
}

#[test]
fn with_exported_function_return_true() {
    let module = atom_unchecked("erlang");
    let function = atom_unchecked("self");
    let arity = 0.into();

    crate::otp::erlang::self_0::export();

    assert_eq!(native(module, function, arity), Ok(true.into()));
}
