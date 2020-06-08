//! ```elixir
//! # label 5
//! # pushed to stack: (value)
//! # returned from call: time
//! # full stack: (time, value)
//! # returns: {time, value}
//! {time, value}
//! ```

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

// Private

#[native_implemented::label]
fn result(process: &Process, time: Term, value: Term) -> exception::Result<Term> {
    assert!(time.is_integer());

    process.tuple_from_slice(&[time, value]).map_err(From::from)
}
