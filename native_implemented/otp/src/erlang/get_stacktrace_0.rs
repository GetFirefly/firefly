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

use firefly_rt::process::{Process, ProcessStatus};
use firefly_rt::term::Term;

#[native_implemented::function(erlang:get_stacktrace/0)]
pub fn result(process: &Process) -> Term {
    match process.status() {
        ProcessStatus::Errored(ref erlang_exception) => erlang_exception.trace().as_term().unwrap(),
        _ => Term::Nil,
    }
}
