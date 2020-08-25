use liblumen_alloc::erts::term::prelude::*;

pub mod apply_4;

pub fn module() -> Atom {
    Atom::from_str("Elixir.Lumen.Web.EventListener")
}

// Private

fn module_id() -> usize {
    module().id()
}
