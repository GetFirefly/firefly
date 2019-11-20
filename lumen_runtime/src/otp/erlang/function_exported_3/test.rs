use liblumen_alloc::atom;

use crate::otp::erlang::function_exported_3::native;
use crate::scheduler::with_process;

#[test]
fn without_exported_function_returns_false() {
    let module = atom!("unexported_module");
    let function = atom!("unexported_function");
    let arity = 0.into();

    with_process(|process| {
        assert_eq!(native(process, module, function, arity), Ok(false.into()));
    });
}

#[test]
fn with_exported_function_return_true() {
    let module = atom!("erlang");
    let function = atom!("self");
    let arity = 0.into();

    crate::otp::erlang::self_0::export();

    with_process(|process| {
        assert_eq!(native(process, module, function, arity), Ok(true.into()));
    });
}
