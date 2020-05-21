use liblumen_alloc::Arity;
use liblumen_alloc::erts::ModuleFunctionArity;
use liblumen_alloc::erts::process::{Frame, Native, Process};
use liblumen_alloc::erts::term::prelude::*;

use crate::process::current_process;

pub const NATIVE: Native = Native::Zero(native);

pub fn function() -> Atom {
    Atom::from_str("loop")
}

pub extern "C" fn native() -> Term {
    let arc_process = current_process();
    arc_process.reduce();

    result(&arc_process)
}

// Private

const ARITY: Arity = 0;

fn frame() -> Frame {
   Frame::new(module_function_arity(), NATIVE)
}

fn module_function_arity() -> ModuleFunctionArity {
    ModuleFunctionArity {
        module: super::module(),
        function: function(),
        arity: ARITY
    }
}

fn result(process: &Process) -> Term {
    process.wait();
    process.queue_frame_with_arguments(frame().with_arguments(false, &[]));

    Term::NONE
}

