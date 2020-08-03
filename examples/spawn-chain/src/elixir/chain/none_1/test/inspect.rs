use liblumen_alloc::erts::term::prelude::*;

use crate::runtime;

#[native_implemented::function(Elixir.ChainTest:inspect/1)]
fn result(time_value: Term) -> Term {
    runtime::sys::io::puts(&format!("{}", time_value));

    Term::NONE
}
