//! ```elixir
//! def counter(next_pid, output) when is_function(output, 1) do
//!   output.("spawned")
//!
//!   receive do
//!     n ->
//!       output.("received #{n}")
//!       sent = send(next_pid, n + 1)
//!       output.("sent #{sent} to #{next_pid}")
//!   end
//! end
//! ```

mod label_1;
mod label_2;
mod label_3;
mod label_4;

use std::convert::TryInto;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

// Private

#[native_implemented::function(Elixir.Chain:counter/2)]
fn result(process: &Process, next_pid: Term, output: Term) -> exception::Result<Term> {
    assert!(next_pid.is_pid());
    // is_function(output, ...)
    let output_closure: Boxed<Closure> = output.try_into().unwrap();

    // is_function(output, 1)
    let output_module_function_arity = output_closure.module_function_arity();
    assert_eq!(output_module_function_arity.arity, 1);

    // ```elixir
    // output.("spawned")
    // ```
    let output_data = process.binary_from_str("spawned")?;
    process
        .queue_frame_with_arguments(output_closure.frame_with_arguments(false, vec![output_data]));

    // # label 1
    // # pushed to stack: (next_pid, output)
    // # returned from called: :ok
    // # full stack: (:ok, next_pid, output)
    // receive do
    //   n ->
    //     output.("received #{n}")
    //     sent = send(next_pid, n + 1)
    //     output.("sent #{sent} to #{next_pid}")
    // end
    process.queue_frame_with_arguments(label_1::frame().with_arguments(true, &[next_pid, output]));

    Ok(Term::NONE)
}
