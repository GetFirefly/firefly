//! Functions in `Lumen.Web.Async` return JS Promises while running the Erlang code in a new
//! `Process`.
//!
//! * The `Promise` is fulfilled if the function returns
//! * The `Promise` is rejected if the `Process` exits abnormally

pub mod apply_3;

use liblumen_alloc::erts::term::prelude::*;

fn module() -> Atom {
    Atom::from_str("Elixir.Lumen.Async")
}

fn module_id() -> usize {
    module().id()
}
