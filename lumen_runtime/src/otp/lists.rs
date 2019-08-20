//! Mirrors [lists](http://erlang.org/doc/man/lists.html) module

pub mod keyfind_3;

use liblumen_alloc::erts::term::Atom;

fn module() -> Atom {
    Atom::try_from_str("lists").unwrap()
}
