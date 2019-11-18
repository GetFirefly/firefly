pub mod reduce_3;
mod reduce_range_dec_4;
mod reduce_range_inc_4;

use liblumen_alloc::erts::term::prelude::Atom;

// Private

fn module() -> Atom {
    Atom::try_from_str("Elixir.Enum").unwrap()
}
