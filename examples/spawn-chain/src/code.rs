use std::convert::TryInto;
use std::sync::Arc;

use lumen_runtime::process::stack::frame::Frame;
use lumen_runtime::process::{ModuleFunctionArity, Process};
use lumen_runtime::term::Term;

use crate::elixir;

pub fn apply(arc_process: &Arc<Process>) {
    let frame_argument_vec = arc_process.pop_arguments(3);
    let module = frame_argument_vec[0];
    let function = frame_argument_vec[1];

    let argument_list = frame_argument_vec[2];
    let argument_vec: Vec<Term> = match argument_list.try_into() {
        Ok(argument_vec) => argument_vec,
        Err(_) => {
            #[cfg(debug_assertions)]
            panic!(
                "Arguments ({:?}) are neither empty list nor proper list",
                argument_list
            );
            #[cfg(not(debug_assertions))]
            panic!("Arguments are neither empty list nor proper list")
        }
    };
    let arity = argument_vec.len();

    match unsafe { module.atom_to_string() }.as_ref().as_ref() {
        "Elixir.Chain" => match unsafe { function.atom_to_string() }.as_ref().as_ref() {
            "counter" => match arity {
                1 => {
                    let module_function_arity = Arc::new(ModuleFunctionArity {
                        module,
                        function,
                        arity,
                    });

                    // Elixir.Chain.counter is a user function and not a BIF, so it is a `Code` and
                    // run in a frame instead of directly
                    let mut chain_counter_frame =
                        Frame::new(module_function_arity, elixir::chain::counter_0_code);
                    chain_counter_frame.push(argument_vec[0]);
                    arc_process.replace_frame(chain_counter_frame);

                    // don't count finding the function as a reduction if it is found, only on
                    // exception in `undef`, so that each path is at least one reduction.
                    Process::call_code(arc_process);
                }
                _ => undef(arc_process, module, function, argument_list),
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
                    let mut chain_run_frame =
                        Frame::new(module_function_arity, elixir::chain::run_0_code);
                    chain_run_frame.push(argument_vec[0]);
                    arc_process.replace_frame(chain_run_frame);

                    // don't count finding the function as a reduction if it is found, only on
                    // exception in `undef`, so that each path is at least one reduction.
                    Process::call_code(arc_process);
                }
                _ => undef(arc_process, module, function, argument_list),
            },
            _ => undef(arc_process, module, function, argument_list),
        },
        _ => undef(arc_process, module, function, argument_list),
    }
}

#[cfg(debug_assertions)]
pub fn print_stacktrace(process: &Process) {
    let mut formatted_stacktrace_parts: Vec<String> = Vec::new();

    for module_function_arity in process.stacktrace() {
        formatted_stacktrace_parts.push(format!("{}", module_function_arity));
    }

    let formatted_stacktrace = formatted_stacktrace_parts.join("\n");

    crate::start::log_1(formatted_stacktrace);
}

fn undef(arc_process: &Arc<Process>, module: Term, function: Term, arguments: Term) {
    arc_process.reduce();
    arc_process.exception(lumen_runtime::undef!(
        module,
        function,
        arguments,
        arc_process
    ))
}
