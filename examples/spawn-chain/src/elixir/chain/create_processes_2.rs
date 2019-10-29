mod label_1;
mod label_2;
mod label_3;

use std::sync::Arc;

use liblumen_alloc::erts::exception::Alloc;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::code::{self, result_from_exception};
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::ModuleFunctionArity;

use crate::elixir;

/// ```elixir
/// def create_processes(n, output) do
///   last =
///     Enum.reduce(
///       1..n,
///       self,
///       fn (_, send_to) ->
///         spawn(Chain, :counter, [send_to, output])
///       end
///     )
///
///   # label 1
///   # pushed stack: (output)
///   # returned from call: last
///   # full stack: (last, output)
///   # returns: sent
///   send(last, 0) # start the count by sending a zero to the last process
///
///   # label 2
///   # pushed to stack: (output)
///   # returned from call: sent
///   # full stack: (sent, output)
///   # returns: :ok
///   receive do # and wait for the result to come back to us
///     final_answer when is_integer(final_answer) ->
///       "Result is #{inspect(final_answer)}"
///   end
/// end
/// ```
pub fn place_frame_with_arguments(
    process: &Process,
    placement: Placement,
    n: Term,
    output: Term,
) -> Result<(), Alloc> {
    assert!(n.is_integer());
    assert!(output.is_function());
    process.stack_push(output)?;
    process.stack_push(n)?;
    process.place_frame(frame(), placement);

    Ok(())
}

// Private

fn code(arc_process: &Arc<Process>) -> code::Result {
    let n = arc_process.stack_pop().unwrap();
    let output = arc_process.stack_pop().unwrap();

    // ```elixir
    // 1..n
    // ```
    // assumed to be fast enough to act as a BIF
    let first = arc_process.integer(1)?;
    let last = n;
    let result = elixir::range::new(first, last, arc_process);

    arc_process.reduce();

    match result {
        Ok(range) => {
            //  # label 1
            //  # pushed stack: (output)
            //  # returned from call: last
            //  # full stack: (last, output)
            //  # returns: sent
            label_1::place_frame_with_arguments(arc_process, Placement::Replace, output)?;

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
            let reducer = elixir::chain::create_processes_reducer_2::closure(arc_process, output)?;
            elixir::r#enum::reduce_3::place_frame_with_arguments(
                arc_process,
                Placement::Push,
                range,
                arc_process.pid_term(),
                reducer,
            )?;

            Process::call_code(arc_process)
        }
        Err(exception) => result_from_exception(arc_process, exception),
    }
}

fn frame() -> Frame {
    Frame::new(module_function_arity(), code)
}

fn function() -> Atom {
    Atom::try_from_str("create_processes").unwrap()
}

fn module_function_arity() -> Arc<ModuleFunctionArity> {
    Arc::new(ModuleFunctionArity {
        module: super::module(),
        function: function(),
        arity: 2,
    })
}
