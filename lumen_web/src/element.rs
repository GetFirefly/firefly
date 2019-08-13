pub mod set_attribute_3;

use liblumen_alloc::erts::term::Atom;

pub fn module() -> Atom {
    Atom::try_from_str("Elixir.Lumen.Web.Element").unwrap()
}
