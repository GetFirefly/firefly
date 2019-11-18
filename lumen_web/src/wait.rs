pub mod with_return_0;

use liblumen_alloc::erts::term::prelude::Atom;

fn module() -> Atom {
    Atom::try_from_str("Elixir.Lumen.Web.Wait").unwrap()
}
