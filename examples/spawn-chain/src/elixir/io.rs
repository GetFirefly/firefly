pub mod puts_1;

use liblumen_alloc::erts::term::prelude::Atom;

fn module() -> Atom {
    Atom::from_str("Elixir.IO")
}

#[allow(dead_code)]
fn module_id() -> usize {
    module().id()
}
