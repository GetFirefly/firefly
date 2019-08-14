pub mod document_1;
pub mod window_0;

use liblumen_alloc::erts::term::Atom;

// Private

fn module() -> Atom {
    Atom::try_from_str("Elixir.Lumen.Web.Window").unwrap()
}
