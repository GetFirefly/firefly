pub mod exit_1;

use liblumen_alloc::erts::term::prelude::*;

fn module() -> Atom {
    Atom::from_str("erlang")
}

