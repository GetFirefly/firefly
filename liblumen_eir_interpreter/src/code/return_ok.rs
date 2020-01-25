use std::sync::Arc;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::code;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::closure::Definition;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::Arity;

use locate_code::locate_code;

#[locate_code]
pub fn code(arc_process: &Arc<Process>) -> code::Result {
    let argument_list = arc_process.stack_pop().unwrap();

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
    assert!(argument_vec.len() == 1);

    Ok(arc_process.return_from_call(0, argument_vec[0])?)
}

pub fn closure(process: &Process) -> exception::Result<Term> {
    let function = Atom::try_from_str("return_ok").unwrap();
    let definition = Definition::Export { function };
    const ARITY: Arity = 1;

    process
        .closure_with_env_from_slice(super::module(), definition, ARITY, Some(LOCATED_CODE), &[])
        .map_err(|error| error.into())
}
