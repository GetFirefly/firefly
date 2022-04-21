//! Mirrors [lists](http://erlang.org/doc/man/lists.html) module

pub mod keyfind_3;
pub mod keymember_3;
pub mod member_2;
pub mod reverse_1;
pub mod reverse_2;

use liblumen_alloc::erts::term::prelude::Atom;

fn module() -> Atom {
    Atom::from_str("lists")
}
