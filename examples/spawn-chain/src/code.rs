use std::convert::TryInto;
use std::sync::Arc;

use liblumen_alloc::erts::exception::runtime;
use liblumen_alloc::erts::process::code::stack::frame::Frame;
use liblumen_alloc::erts::process::code::Result;
use liblumen_alloc::erts::process::ProcessControlBlock;
use liblumen_alloc::erts::term::{Atom, Term, TypedTerm};
use liblumen_alloc::erts::ModuleFunctionArity;

use crate::elixir;

pub fn apply(arc_process: &Arc<ProcessControlBlock>) -> Result {
    let module_term = arc_process.stack_pop().unwrap();
    let function_term = arc_process.stack_pop().unwrap();

    let argument_list = arc_process.stack_pop().unwrap();

    let mut argument_vec: Vec<Term> = Vec::new();

    match argument_list.to_typed_term().unwrap() {
        TypedTerm::Nil => (),
        TypedTerm::List(argument_cons) => {
            for result in argument_cons.into_iter() {
                let element = result.unwrap();

                argument_vec.push(element);
            }
        }
        _ => panic!(
            "In {:?}, {:?} ({:#b}) is not an argument list.  Cannot call {:?}:{:?}",
            arc_process.pid_term(),
            argument_list,
            argument_list.as_usize(),
            module_term,
            function_term
        ),
    }

    let module: Atom = module_term.try_into().unwrap();
    let function: Atom = function_term.try_into().unwrap();
    let arity = argument_vec.len() as u8;

    match module.name() {
        "Elixir.Chain" => match function.name() {
            "counter" => match arity {
                1 => {
                    let module_function_arity = Arc::new(ModuleFunctionArity {
                        module,
                        function,
                        arity,
                    });

                    // Elixir.Chain.counter is a user function and not a BIF, so it is a `Code` and
                    // run in a frame instead of directly
                    let chain_counter_frame =
                        Frame::new(module_function_arity, elixir::chain::counter_0_code);
                    arc_process.stack_push(argument_vec[0])?;
                    arc_process.replace_frame(chain_counter_frame);

                    // don't count finding the function as a reduction if it is found, only on
                    // exception in `undef`, so that each path is at least one reduction.
                    ProcessControlBlock::call_code(arc_process)
                }
                _ => undef(arc_process, module_term, function_term, argument_list),
            },
            "run" => match arity {
                1 => {
                    let module_function_arity = Arc::new(ModuleFunctionArity {
                        module,
                        function,
                        arity,
                    });
                    // Elixir.Chain.run is a user function and not a BIF, so it is a `Code` and
                    // run in a frame instead of directly
                    let chain_run_frame =
                        Frame::new(module_function_arity, elixir::chain::run_0_code);
                    arc_process.stack_push(argument_vec[0])?;
                    arc_process.replace_frame(chain_run_frame);

                    // don't count finding the function as a reduction if it is found, only on
                    // exception in `undef`, so that each path is at least one reduction.
                    ProcessControlBlock::call_code(arc_process)
                }
                _ => undef(arc_process, module_term, function_term, argument_list),
            },
            _ => undef(arc_process, module_term, function_term, argument_list),
        },
        _ => undef(arc_process, module_term, function_term, argument_list),
    }
}

fn undef(
    arc_process: &Arc<ProcessControlBlock>,
    module: Term,
    function: Term,
    arguments: Term,
) -> Result {
    arc_process.reduce();
    let exception = liblumen_alloc::undef!(arc_process, module, function, arguments);
    let runtime_exception: runtime::Exception = exception.try_into().unwrap();
    arc_process.exception(runtime_exception);

    Ok(())
}
