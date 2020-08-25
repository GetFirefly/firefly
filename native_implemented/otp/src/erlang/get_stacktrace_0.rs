//! ```elixir
//! @spec get_stacktrace() :: [stack_item()]
//!   when stack_time: {module :: atom(),
//!                     function :: atom(),
//!                     arity :: 0..255 | (args :: [term()]),
//!                     location :: [{:file, filename :: charlist()} |
//!                                  {:line, line :: pos_integer()}]}
//! ```

#[cfg(test)]
mod test;

use liblumen_alloc::erts::process::{Process, Status};
use liblumen_alloc::erts::term::prelude::Term;

#[native_implemented::function(erlang:get_stacktrace/0)]
pub fn result(process: &Process) -> Term {
    match *process.status.read() {
        Status::RuntimeException(ref exc) => {
            if let Some(trace) = exc.stacktrace() {
                trace.as_term().unwrap()
            } else {
                Term::NIL
            }
        }
        _ => Term::NIL,
    }
}
