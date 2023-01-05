#[cfg(test)]
mod test;

use firefly_rt::term::Term;

use crate::runtime::distribution::nodes::node;

#[native_implemented::function(erlang:node/0)]
pub fn result() -> Term {
    node::term()
}
