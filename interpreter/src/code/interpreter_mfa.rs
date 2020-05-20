use std::sync::Arc;

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::exec::CallExecutor;

use native_implemented_function::native_implemented_function;

/// Expects the following on stack:
/// * arity integer
/// * argument list
#[native_implemented_function(interpreter_mfa/1)]
pub fn result(arc_process: Arc<Process>, argument_list: Term) -> Term {
    let mfa = arc_process.current_module_function_arity().unwrap();

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
    assert!(mfa.arity as usize == argument_vec.len() - 2);

    let mut exec = CallExecutor::new();
    exec.call(
        &crate::VM,
        &arc_process,
        mfa.module,
        mfa.function,
        argument_vec.len() - 2,
        &mut argument_vec,
    );

    Term::NONE
}
