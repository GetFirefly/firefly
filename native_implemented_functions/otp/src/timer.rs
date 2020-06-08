use liblumen_alloc::erts::term::prelude::Atom;

pub mod tc_3;

pub mod cancel;
pub mod read;
pub mod start;

fn module() -> Atom {
    Atom::try_from_str("timer").unwrap()
}

fn module_id() -> usize {
    module().id()
}
