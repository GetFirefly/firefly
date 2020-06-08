//! ```elixir
//! # pushed to stack: (n)
//! # returned from call: N/A
//! # full stack: (n)
//! # returns: final_answer
//! def console(n) do
//!   run(n, &console_output/1)
//! end
//! ```

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::elixir::chain::{console_output_1, run_2};

// Private

#[native_implemented::function(console/1)]
fn result(process: &Process, n: Term) -> exception::Result<Term> {
    assert!(n.is_integer());

    let console_output_closure = console_output_1::closure(process)?;
    process.queue_frame_with_arguments(
        run_2::frame().with_arguments(false, &[n, console_output_closure]),
    );

    Ok(Term::NONE)
}
