use std::ffi::c_void;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use native_implemented_function::native_implemented_function;

pub fn closure(process: &Process) -> exception::Result<Term> {
    let function = Atom::try_from_str("return_ok").unwrap();
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

#[native_implemented_function(return_ok/1)]
pub fn result(argument_list: Term) -> Term {
    let mut argument_vec: Vec<Term> = Vec::new();
    match argument_list.decode().unwrap() {
        TypedTerm::Nil => (),
        TypedTerm::List(argument_cons) => {
            for result in argument_cons.into_iter() {
                let element = result.unwrap();

                argument_vec.push(element);
            }
        }
        _ => panic!(),
    }
    assert_eq!(argument_vec.len(), 1);

    argument_vec[0]
}
