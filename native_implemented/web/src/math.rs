pub mod random_integer_1;

use liblumen_alloc::erts::term::prelude::Atom;

fn module() -> Atom {
    Atom::try_from_str("Elixir.Lumen.Web.HTMLFormElement").unwrap()
}

fn module_id() -> usize {
    module().id()
}
