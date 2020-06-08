#[path = "without_body/label_1.rs"]
pub mod label_1;

use super::*;

fn frame_for_native(native: Native) -> Frame {
    Frame::new(module_function_arity(), native)
}

fn function() -> Atom {
    Atom::from_str("body_1_without_body")
}

fn module_function_arity() -> ModuleFunctionArity {
    ModuleFunctionArity {
        module: super::module(),
        function: function(),
        arity: super::ARITY,
    }
}
