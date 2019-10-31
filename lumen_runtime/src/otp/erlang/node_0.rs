#[cfg(test)]
mod test;

use liblumen_alloc::erts::term::Term;

use lumen_runtime_macros::native_implemented_function;

use crate::distribution::nodes::node;

#[native_implemented_function(node/0)]
pub fn native() -> Term {
    node::term()
}
