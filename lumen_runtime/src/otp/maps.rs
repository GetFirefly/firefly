pub mod find_2;
pub mod get_3;
pub mod is_key_2;
pub mod merge_2;

use liblumen_alloc::erts::term::Atom;

fn module() -> Atom {
    Atom::try_from_str("maps").unwrap()
}
