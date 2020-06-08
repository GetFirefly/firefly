//! ```elixir
//! # pushed to stack: (n)
//! # returned from call: N/A
//! # full stack: (n)
//! # returns: final_answer
//! def dom(n) do
//!   run(n, &dom_output/1)
//! end
//! ```

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::elixir::chain::{dom_output_1, run_2};

// Private

#[native_implemented::function(dom/1)]
fn result(process: &Process, n: Term) -> exception::Result<Term> {
    assert!(n.is_integer());

    let dom_output_closure = dom_output_1::closure(process)?;
    process
        .queue_frame_with_arguments(run_2::frame().with_arguments(false, &[n, dom_output_closure]));

    Ok(Term::NONE)
}
