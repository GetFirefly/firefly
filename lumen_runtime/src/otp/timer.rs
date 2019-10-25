use liblumen_alloc::erts::term::prelude::Atom;

pub mod tc_3;

fn module() -> Atom {
    Atom::try_from_str("timer").unwrap()
}
