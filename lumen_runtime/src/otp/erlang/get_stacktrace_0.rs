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

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::exception::Stacktrace;
use liblumen_alloc::erts::process::code::stack::Trace;
use liblumen_alloc::erts::process::{Process, Status};
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::ModuleFunctionArity;

use lumen_runtime_macros::native_implemented_function;

#[native_implemented_function(get_stacktrace/0)]
pub fn native(process: &Process) -> exception::Result<Term> {
    match *process.status.read() {
        Status::Exiting(ref runtime_exception) => match runtime_exception.stacktrace() {
            Some(stacktrace) => match stacktrace {
                Stacktrace::Trace(Trace(module_function_arity_vec)) => {
                    let mut stack_item_vec: Vec<Term> =
                        Vec::with_capacity(module_function_arity_vec.len());

                    for module_function_arity in module_function_arity_vec {
                        let stack_item = stack_item(process, &module_function_arity)?;
                        stack_item_vec.push(stack_item);
                    }

                    process
                        .list_from_slice(&stack_item_vec)
                        .map_err(|error| error.into())
                }
                Stacktrace::Term(term) => Ok(term),
            },
            None => Ok(Term::NIL),
        },
        _ => Ok(Term::NIL),
    }
}

fn stack_item(
    process: &Process,
    ModuleFunctionArity {
        module,
        function,
        arity,
    }: &ModuleFunctionArity,
) -> exception::Result<Term> {
    let module_term = module.encode()?;
    let function_term = function.encode()?;
    let arity_term = process.integer(*arity)?;

    process
        .tuple_from_slice(&[module_term, function_term, arity_term])
        .map_err(|error| error.into())
}
