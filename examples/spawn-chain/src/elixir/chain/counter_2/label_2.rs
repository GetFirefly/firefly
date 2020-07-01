use std::convert::TryInto;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use liblumen_otp::erlang;

use super::label_3;

#[native_implemented::label]
fn result(process: &Process, n: Term, next_pid: Term, output: Term) -> exception::Result<Term> {
    assert!(n.is_integer());
    assert!(next_pid.is_pid());
    let _: Boxed<Closure> = output.try_into().unwrap();

    // ```elixir
    // # pushed to stack: (n)
    // # return from call: N/A
    // # full stack: (n)
    // # returns: sum
    // n + 1
    let one = process.integer(1)?;
    process.queue_frame_with_arguments(erlang::add_2::frame().with_arguments(false, &[n, one]));

    // ```elixir
    // # label 3
    // # pushed stack: (next_pid, output)
    // # returned from call: sum
    // # full stack: (sum, next_pid, output)
    // # returns: sent
    // sent = send(next_pid, sum)
    // output.("send #{sent} to #{next_pid}")
    // ```
    process.queue_frame_with_arguments(label_3::frame().with_arguments(true, &[next_pid, output]));

    Ok(Term::NONE)
}
