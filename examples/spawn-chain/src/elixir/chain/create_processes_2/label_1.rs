//! ```elixir
//! # label 1
//! # pushed stack: (output)
//! # returned from call: last
//! # full stack: (last, output)
//! # returns: sent
//! send(last, 0) # start the count by sending a zero to the last process
//!
//! # label 2
//! # pushed to stack: (output)
//! # returned from call: sent
//! # full stack: (sent, output)
//! # returns: :ok
//! receive do # and wait for the result to come back to us
//!   final_answer when is_integer(final_answer) ->
//!     output.("Result is #{inspect(final_answer)}")
//!     final_answer
//! end
//! ```

use std::convert::TryInto;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use liblumen_otp::erlang;

use super::label_2;

#[native_implemented::label]
fn result(process: &Process, last: Term, output: Term) -> exception::Result<Term> {
    // placed on top of stack by return from `elixir::r#enum::reduce_0_code`
    assert!(last.is_local_pid(), "last ({:?}) is not a local pid", last);
    let output_closure: Boxed<Closure> = output.try_into().unwrap();
    assert_eq!(output_closure.arity(), 1);

    // ```elixir
    // # pushed stack: (last, data)
    // # returned from call: N/A
    // # full stack: (last, data)
    // # returns: sent
    // send(last, 0) # start the count by sending a zero to the last process
    // ```
    let message = process.integer(0).unwrap();
    process.queue_frame_with_arguments(
        erlang::send_2::frame().with_arguments(false, &[last, message]),
    );

    // ```elixir
    // # label 2
    // # pushed to stack: (output)
    // # returned from call: sent
    // # full stack: (sent, output)
    // # returns: :ok
    // receive do # and wait for the result to come back to us
    //   final_answer when is_integer(final_answer) ->
    //     output.("Result is #{inspect(final_answer)}")
    // end
    // ```
    process.queue_frame_with_arguments(label_2::frame().with_arguments(true, &[output]));

    Ok(Term::NONE)
}
