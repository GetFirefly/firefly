#[cfg(test)]
mod test;

use liblumen_alloc::erts::term::{atom_unchecked, Term};

use lumen_runtime_macros::native_implemented_function;

use crate::node;

#[native_implemented_function(node/0)]
pub fn native() -> Term {
    atom_unchecked(node::DEAD)
}
