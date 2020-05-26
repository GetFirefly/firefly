#[path = "with_body/label_1.rs"]
pub mod label_1;
#[path = "with_body/label_2.rs"]
pub mod label_2;

use super::*;

fn frame(native: Native) -> Frame {
    Frame::new(module_function_arity(), native)
}

fn function() -> Atom {
    Atom::from_str("body_1_with_body")
}

fn module_function_arity() -> ModuleFunctionArity {
    ModuleFunctionArity {
        module: super::module(),
        function: function(),
        arity: super::ARITY,
    }
}
