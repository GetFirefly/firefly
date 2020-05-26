#[path = "with_reference_child_inserts_before_reference_child/label_1.rs"]
pub mod label_1;
#[path = "with_reference_child_inserts_before_reference_child/label_2.rs"]
pub mod label_2;
#[path = "with_reference_child_inserts_before_reference_child/label_3.rs"]
pub mod label_3;
#[path = "with_reference_child_inserts_before_reference_child/label_4.rs"]
pub mod label_4;
#[path = "with_reference_child_inserts_before_reference_child/label_5.rs"]
pub mod label_5;
#[path = "with_reference_child_inserts_before_reference_child/label_6.rs"]
pub mod label_6;

use liblumen_alloc::erts::process::{Frame, Native};
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::{Arity, ModuleFunctionArity};

const ARITY: Arity = 0;

fn frame(native: Native) -> Frame {
    Frame::new(module_function_arity(), native)
}

fn function() -> Atom {
    Atom::from_str("insert_before_3_with_reference_child_inserts_before_reference_child")
}

fn module() -> Atom {
    Atom::from_str("Lumen.Web.NodeTest")
}

fn module_function_arity() -> ModuleFunctionArity {
    ModuleFunctionArity {
        module: module(),
        function: function(),
        arity: ARITY,
    }
}
