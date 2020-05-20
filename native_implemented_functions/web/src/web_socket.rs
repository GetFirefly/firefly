pub mod new_1;

use liblumen_alloc::erts::term::prelude::Atom;

pub fn module() -> Atom {
    Atom::try_from_str("Elixir.Lumen.Web.WebSocket").unwrap()
}

fn module_id() -> usize {
    module().id()
}
