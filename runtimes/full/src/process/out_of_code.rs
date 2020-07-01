use anyhow::*;

use liblumen_alloc::erts::process::{Frame, Native};
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::{Arity, ModuleFunctionArity};

use lumen_rt_core::process::current_process;

pub fn frame() -> Frame {
    Frame::new(module_function_arity(), Native::Zero(native))
}

// Private

const ARITY: Arity = 0;

fn function() -> Atom {
    Atom::try_from_str("lumen").unwrap()
}

fn module() -> Atom {
    Atom::try_from_str("out_of_code").unwrap()
}

fn module_function_arity() -> ModuleFunctionArity {
    ModuleFunctionArity {
        module: module(),
        function: function(),
        arity: ARITY,
    }
}

extern "C" fn native() -> Term {
    current_process().exit_normal(anyhow!("Out of code").into());

    Term::NONE
}
