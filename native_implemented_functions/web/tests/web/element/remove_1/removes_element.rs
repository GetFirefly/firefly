#[path = "removes_element/label_1.rs"]
pub mod label_1;
#[path = "removes_element/label_2.rs"]
pub mod label_2;
#[path = "removes_element/label_3.rs"]
pub mod label_3;
#[path = "removes_element/label_4.rs"]
pub mod label_4;
#[path = "removes_element/label_5.rs"]
pub mod label_5;

use liblumen_alloc::erts::process::{Frame, Native};
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::{Arity, ModuleFunctionArity};

const ARITY: Arity = 1;

fn frame(native: Native) -> Frame {
    Frame::new(module_function_arity(), native)
}

fn function() -> Atom {
    Atom::from_str("remove_1_removes_element")
}

fn module() -> Atom {
    Atom::from_str("Lumen.Web.ElementTest")
}

fn module_function_arity() -> ModuleFunctionArity {
    ModuleFunctionArity {
        module: module(),
        function: function(),
        arity: ARITY,
    }
}
