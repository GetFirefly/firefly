//! ```elixir
//! # label 3
//! # pushed to stack: (next_pid, output)
//! # returned from call: sum
//! # full stack: (sum, next_pid, output)
//! # returns: sent
//! sent = send(next_pid, sum)
//! output.("sent #{sent} to #{next_pid}")
//! ```

use std::convert::TryInto;

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use liblumen_otp::erlang;

use super::label_4;

// Private

#[native_implemented::label]
fn result(process: &Process, sum: Term, next_pid: Term, output: Term) -> Term {
    assert!(sum.is_integer());
    assert!(next_pid.is_pid());
    let _: Boxed<Closure> = output.try_into().unwrap();

    // ```elixir
    // # pushed stack: (next_pid, sum)
    // # returned from call: N/A
    // # full stack: (next_pid, sum)
    // # returns: sent
    // send(next_pid, sum)
    process.queue_frame_with_arguments(
        erlang::send_2::frame().with_arguments(false, &[next_pid, sum]),
    );

    // ```elixir
    // # label 4
    // # pushed stack: (output, next_pid)
    // # returned from call: sent
    // # full stack: (sent, output, next_pid)
    // # returns: :ok
    // sent = ...
    // output.("sent #{sent} to #{next_pid}")
    // ```
    process.queue_frame_with_arguments(label_4::frame().with_arguments(true, &[output, next_pid]));

    Term::NONE
}
