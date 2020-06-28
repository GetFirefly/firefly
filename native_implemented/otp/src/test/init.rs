pub mod start_0;

use liblumen_alloc::erts::term::prelude::*;

fn module() -> Atom {
    Atom::from_str("init")
}

fn module_id() -> usize {
    module().id()
}
