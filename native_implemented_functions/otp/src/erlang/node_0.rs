#[cfg(test)]
mod test;

use liblumen_alloc::erts::term::prelude::*;

use crate::runtime::distribution::nodes::node;

use native_implemented_function::native_implemented_function;

#[native_implemented_function(node/0)]
pub fn result() -> Term {
    node::term()
}
