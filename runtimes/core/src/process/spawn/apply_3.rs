use std::ffi::c_void;
use std::mem::transmute;

use liblumen_core::sys::dynamic_call::DynamicCallee;

use liblumen_alloc::erts::apply::find_symbol;
use liblumen_alloc::erts::process::{Frame, Native};
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::{Arity, ModuleFunctionArity};

pub fn frame() -> Frame {
    let module_function_arity = module_function_arity();
    let native = find_symbol(&module_function_arity)
        .map(|dynamic_callee| unsafe { transmute::<DynamicCallee, *const c_void>(dynamic_callee) })
        .expect("erlang:apply/3 not exported");

    Frame::new(module_function_arity, unsafe {
        Native::from_ptr(native, ARITY)
    })
}

// Private

const ARITY: Arity = 3;

fn module_function_arity() -> ModuleFunctionArity {
    ModuleFunctionArity {
        module: Atom::from_str("erlang"),
        function: Atom::from_str("apply"),
        arity: ARITY,
    }
}
