pub mod interpreter_closure;
pub mod interpreter_mfa;
pub mod return_clean;
pub mod return_ok;
pub mod return_throw;

use std::convert::TryInto;

use liblumen_alloc::erts::process::Frame;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::erts::ModuleFunctionArity;

use crate::runtime::process::current_process;

fn module() -> Atom {
    Atom::try_from_str("lumen_eir_interpreter_intrinsics").unwrap()
}

fn module_id() -> usize {
    module().id()
}

pub extern "C" fn apply(module_term: Term, function_term: Term, argument_list: Term) -> Term {
    let module: Atom = module_term.try_into().unwrap();
    let function: Atom = function_term.try_into().unwrap();

    let arity;
    match argument_list.decode().unwrap() {
        TypedTerm::Nil => panic!(),
        TypedTerm::List(argument_cons) => arity = argument_cons.into_iter().count() - 2,
        _ => panic!(),
    }

    let module_function_arity = ModuleFunctionArity {
        module,
        function,
        arity: arity.try_into().unwrap(),
    };

    let frame = Frame::new(module_function_arity, interpreter_mfa::NATIVE);
    let frame_with_arguments = frame.with_arguments(false, &[argument_list]);

    current_process().queue_frame_with_arguments(frame_with_arguments);

    Term::NONE
}
