use std::ffi::c_void;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

pub fn closure(process: &Process) -> exception::Result<Term> {
    let function = Atom::try_from_str("return_clean").unwrap();
    const ARITY: u8 = 1;

    process
        .export_closure(
            super::module(),
            function,
            ARITY,
            Some(native as *const c_void),
        )
        .map_err(|error| error.into())
}

#[native_implemented::function(lumen_eir_interpreter_intrinsics:return_clean/1)]
pub fn result(argument_list: Term) -> Term {
    argument_list
}
