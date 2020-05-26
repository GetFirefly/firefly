use anyhow::*;

use liblumen_alloc::erts::process::{Frame, Native};
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::erts::{exception, ModuleFunctionArity};
use liblumen_alloc::{exit, Arity};

pub fn frame() -> Frame {
    Frame::new(module_function_arity(), NATIVE)
}

pub extern "C" fn native(reason: Term) -> Term {
    let arc_process = crate::process::current_process();
    arc_process.reduce();

    arc_process.return_status(result(reason))
}

// Private

const ARITY: Arity = 1;
const NATIVE: Native = Native::One(native);

fn function() -> Atom {
    Atom::from_str("exit")
}

fn module_function_arity() -> ModuleFunctionArity {
    ModuleFunctionArity {
        module: super::module(),
        function: function(),
        arity: ARITY,
    }
}

fn result(reason: Term) -> exception::Result<Term> {
    Err(exit!(reason, anyhow!("explicit exit from Erlang").into()).into())
}
