//! ```elixir
//! # label 2
//! # pushed to stack: ({time, value})
//! # returned from call: :ok
//! # full stack: (:ok, {time, value})
//! # returns: {time, value}
//! {time, value}

use liblumen_alloc::atom;
use liblumen_alloc::erts::term::prelude::*;

// Private

#[native_implemented::label]
fn result(ok: Term, time_value: Term) -> Term {
    assert_eq!(ok, atom!("ok"));

    time_value
}
