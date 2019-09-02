pub mod find_2;
pub mod get_3;
pub mod is_key_2;
pub mod keys_1;
pub mod merge_2;
pub mod values_1;

use liblumen_alloc::erts::term::Atom;

fn module() -> Atom {
    Atom::try_from_str("maps").unwrap()
}
