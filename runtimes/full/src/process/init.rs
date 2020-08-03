use liblumen_alloc::erts::process::Native;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::{Arity, ModuleFunctionArity};
use lumen_rt_core::process::current_process;

pub const NATIVE: Native = Native::Zero(native);

pub fn module_function_arity() -> ModuleFunctionArity {
    ModuleFunctionArity {
        module: module(),
        function: function(),
        arity: ARITY,
    }
}

// Private

const ARITY: Arity = 0;

fn function() -> Atom {
    Atom::from_str("start")
}

fn module() -> Atom {
    Atom::from_str("init")
}

extern "C" fn native() -> Term {
    current_process().wait();

    Term::NONE
}
