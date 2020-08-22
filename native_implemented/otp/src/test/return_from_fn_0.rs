use liblumen_alloc::erts::term::prelude::*;

pub fn returned() -> Term {
    Atom::str_to_term("returned_from_fn")
}

#[native_implemented::function(test:return_from_fn/0)]
fn result() -> Term {
    returned()
}
