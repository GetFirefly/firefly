use std::sync::Arc;

use liblumen_alloc::erts::process::code;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::closure::Definition;
use liblumen_alloc::erts::term::prelude::*;

use locate_code::locate_code;

use crate::exec::CallExecutor;

/// Expects the following on stack:
/// * arity integer
/// * argument list
#[locate_code]
pub fn code(arc_process: &Arc<Process>) -> code::Result {
    let argument_list = arc_process.stack_pop().unwrap();

    let frame = arc_process.current_frame().unwrap();
    let function = match frame.definition {
        Definition::Export { function } => function,
        definition => unimplemented!("Cannot convert {:?} to function", definition),
    };

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
    assert_eq!(frame.arity as usize, argument_vec.len() - 2);

    let mut exec = CallExecutor::new();
    exec.call(
        &crate::VM,
        arc_process,
        frame.module,
        function,
        argument_vec.len() - 2,
        &mut argument_vec,
    );

    Ok(())
}
