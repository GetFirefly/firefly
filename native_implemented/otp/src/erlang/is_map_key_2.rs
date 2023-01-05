use firefly_rt::*;

use crate::maps;

pub fn frame() -> Frame {
    Frame::new(module_function_arity(), maps::is_key_2::NATIVE)
}

// Private

const ARITY: Arity = 2;

fn function() -> Atom {
    Atom::try_from_str("is_map_key").unwrap()
}

fn module_function_arity() -> ModuleFunctionArity {
    ModuleFunctionArity {
        module: super::module(),
        function: function(),
        arity: ARITY,
    }
}
