#[cfg(test)]
mod test;

use liblumen_alloc::erts::term::prelude::*;

use native_implemented_function::native_implemented_function;

use lumen_runtime::distribution::nodes::node;

#[native_implemented_function(node/0)]
pub fn native() -> Term {
    node::term()
}
