//! ```elixir
//! # pushed to stack: (n)
//! # returned from call: N/A
//! # full stack: (n)
//! # returns: final_answer
//! def none(n) do
//!   run(n, &none_output/1)
//! end
//! ```

#[cfg(test)]
mod test;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::elixir::chain::{none_output_1, run_2};

// Private

#[native_implemented::function(none/1)]
fn result(process: &Process, n: Term) -> exception::Result<Term> {
    assert!(n.is_integer());

    let none_output_closure = none_output_1::closure(process)?;
    process.queue_frame_with_arguments(
        run_2::frame().with_arguments(false, &[n, none_output_closure]),
    );

    Ok(Term::NONE)
}
