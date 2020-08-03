//! ```elixir
//! # label 4
//! # pushed stack: (output, next_pid)
//! # returned from call: sent
//! # full stack: (sent, output, next_pid)
//! # returns: :ok
//! sent = ...
//! output.("sent #{sent} to #{next_pid}")
//! ```

use std::convert::TryInto;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

#[native_implemented::label]
fn result(process: &Process, sent: Term, output: Term, next_pid: Term) -> exception::Result<Term> {
    let output_closure: Boxed<Closure> = output.try_into().unwrap();
    assert_eq!(output_closure.arity(), 1);

    // TODO use `<>` and `to_string` instead of `format!` to properly emulate interpolation
    let data = process.binary_from_str(&format!("sent {} to {}", sent, next_pid))?;
    process.queue_frame_with_arguments(output_closure.frame().with_arguments(false, &[data]));

    Ok(Term::NONE)
}
