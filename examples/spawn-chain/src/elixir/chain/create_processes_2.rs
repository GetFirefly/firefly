//! ```elixir
//! def create_processes(n, output) do
//!   last =
//!     Enum.reduce(
//!       1..n,
//!       self,
//!       fn (_, send_to) ->
//!         spawn(Chain, :counter, [send_to, output])
//!       end
//!     )
//!
//!   # label 1
//!   # pushed stack: (output)
//!   # returned from call: last
//!   # full stack: (last, output)
//!   # returns: sent
//!   send(last, 0) # start the count by sending a zero to the last process
//!
//!   # label 2
//!   # pushed to stack: (output)
//!   # returned from call: sent
//!   # full stack: (sent, output)
//!   # returns: :ok
//!   receive do # and wait for the result to come back to us
//!     final_answer when is_integer(final_answer) ->
//!       "Result is #{inspect(final_answer)}"
//!   end
//! end
//! ```

mod label_1;
mod label_2;
mod label_3;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::elixir;

#[native_implemented::function(create_processes/2)]
fn result(process: &Process, n: Term, output: Term) -> exception::Result<Term> {
    // ```elixir
    // 1..n
    // ```
    // assumed to be fast enough to act as a BIF
    let first = process.integer(1)?;
    let last = n;
    let result = elixir::range::new(first, last, process);

    match result {
        Ok(range) => {
            // ```elixir
            // # returns: last
            // Enum.reduce(
            //    1..n,
            //    self,
            //    fn (_, send_to) ->
            //      spawn(Chain, :counter, [send_to, output])
            //    end
            //  )
            // ```
            let reducer = elixir::chain::create_processes_reducer_3::closure(process, output)?;
            process.queue_frame_with_arguments(
                elixir::r#enum::reduce_3::frame()
                    .with_arguments(false, &[range, process.pid_term(), reducer]),
            );

            //  # label 1
            //  # pushed stack: (output)
            //  # returned from call: last
            //  # full stack: (last, output)
            //  # returns: sent
            process.queue_frame_with_arguments(label_1::frame().with_arguments(true, &[output]));

            Ok(Term::NONE)
        }
        Err(exception) => Err(exception),
    }
}
