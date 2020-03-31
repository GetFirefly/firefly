#[cfg(test)]
mod test;

use liblumen_alloc::erts::term::prelude::*;

use native_implemented_function::native_implemented_function;

use lumen_rt_full::distribution::nodes::node;

#[native_implemented_function(node/0)]
pub fn native() -> Term {
    node::term()
}
