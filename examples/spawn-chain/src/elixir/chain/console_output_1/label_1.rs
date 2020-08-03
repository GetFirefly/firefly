//! ```elixir
//! # label 1
//! # pushed to stack: (text)
//! # returned from call: self
//! # full stack: (self, text)
//! # returns: :ok
//! IO.puts("#{self()} #{text}")
//! ```

use std::convert::TryInto;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::{Pid, Term};

use crate::elixir;

#[native_implemented::label]
fn result(process: &Process, self_pid_term: Term, text: Term) -> exception::Result<Term> {
    let self_pid: Pid = self_pid_term.try_into().unwrap();

    // TODO use `<>` and `to_string` to emulate interpolation properly
    let full_text = process.binary_from_str(&format!("pid={} {}", self_pid, text))?;
    process.queue_frame_with_arguments(
        elixir::io::puts_1::frame().with_arguments(false, &[full_text]),
    );

    Ok(Term::NONE)
}
