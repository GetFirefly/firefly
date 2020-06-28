pub mod find_2;
pub mod from_list_1;
pub mod get_2;
pub mod get_3;
pub mod is_key_2;
pub mod keys_1;
pub mod merge_2;
pub mod put_3;
pub mod remove_2;
pub mod take_2;
pub mod update_3;
pub mod values_1;

use liblumen_alloc::erts::term::prelude::Atom;

fn module() -> Atom {
    Atom::from_str("maps")
}

fn module_id() -> usize {
    module().id()
}
