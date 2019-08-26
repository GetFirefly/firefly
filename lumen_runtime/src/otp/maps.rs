pub mod is_key_2;

use liblumen_alloc::erts::term::Atom;

fn module() -> Atom {
    Atom::try_from_str("maps").unwrap()
}
