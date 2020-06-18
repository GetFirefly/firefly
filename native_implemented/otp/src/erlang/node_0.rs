#[cfg(test)]
mod test;

use liblumen_alloc::erts::term::prelude::*;

use crate::runtime::distribution::nodes::node;

#[native_implemented::function(erlang:node/0)]
pub fn result() -> Term {
    node::term()
}
