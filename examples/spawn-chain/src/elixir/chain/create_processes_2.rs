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

use std::sync::Arc;

use liblumen_alloc::erts::process::frames::stack::frame::Placement;
use liblumen_alloc::erts::process::frames::{self, exception_to_native_return};
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::Arity;

use crate::elixir;

pub fn export() {
    lumen_rt_full::code::export::insert(super::module(), function(), ARITY, code);
}

// Private

const ARITY: Arity = 2;

fn code(arc_process: &Arc<Process>) -> frames::Result {
    let n = arc_process.stack_peek(1).unwrap();
    let output = arc_process.stack_peek(2).unwrap();

    // ```elixir
    // 1..n
    // ```
    // assumed to be fast enough to act as a BIF
    let first = arc_process.integer(1).unwrap();
    let last = n;
    let result = elixir::range::new(first, last, arc_process);

    arc_process.reduce();

    const STACK_USED: usize = 2;

    match result {
        Ok(range) => {
            arc_process.stack_popn(STACK_USED);

            //  # label 1
            //  # pushed stack: (output)
            //  # returned from call: last
            //  # full stack: (last, output)
            //  # returns: sent
            label_1::place_frame_with_arguments(arc_process, Placement::Replace, output).unwrap();

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
            let reducer =
                elixir::chain::create_processes_reducer_2::closure(arc_process, output).unwrap();
            elixir::r#enum::reduce_3::place_frame_with_arguments(
                arc_process,
                Placement::Push,
                range,
                arc_process.pid_term(),
                reducer,
            )
            .unwrap();

            Process::call_native_or_yield(arc_process)
        }
        Err(exception) => exception_to_native_return(arc_process, STACK_USED, exception),
    }
}

fn function() -> Atom {
    Atom::try_from_str("create_processes").unwrap()
}
