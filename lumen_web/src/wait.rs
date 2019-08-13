pub mod with_return_0;

use liblumen_alloc::erts::term::Atom;

fn module() -> Atom {
    Atom::try_from_str("Elixir.Lumen.Web.Wait").unwrap()
}
