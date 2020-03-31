#[cfg(test)]
mod test;

use liblumen_alloc::erts::term::prelude::*;

use lumen_rt_core::distribution::nodes::node;

use native_implemented_function::native_implemented_function;

#[native_implemented_function(node/0)]
pub fn native() -> Term {
    node::term()
}
