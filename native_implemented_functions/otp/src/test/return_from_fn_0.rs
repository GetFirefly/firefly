use std::ffi::c_void;

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use native_implemented_function::native_implemented_function;

pub fn export_closure(process: &Process) -> Term {
    process
        .export_closure(
            super::module(),
            function(),
            ARITY,
            Some(native as *const c_void),
        )
        .unwrap()
}

pub fn returned() -> Term {
    Atom::str_to_term("returned_from_fn")
}

#[native_implemented_function(return_from_fn/0)]
fn result() -> Term {
    returned()
}
