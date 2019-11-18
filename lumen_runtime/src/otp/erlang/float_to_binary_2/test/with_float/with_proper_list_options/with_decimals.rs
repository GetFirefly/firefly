mod with_compact;
mod without_compact;

use super::*;

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Atom;

fn option(process: &Process, digits: u8) -> Term {
    process
        .tuple_from_slice(&[tag(), process.integer(digits).unwrap()])
        .unwrap()
}

fn tag() -> Term {
    Atom::str_to_term("decimals")
}
