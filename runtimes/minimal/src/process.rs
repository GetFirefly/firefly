use liblumen_core::sys::dynamic_call::DynamicCallee;

use liblumen_alloc::erts;
use liblumen_alloc::erts::process::Native;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::ModuleFunctionArity;

pub use lumen_rt_core::process::{current_process, monitor, replace_log_exit, set_log_exit, spawn};

#[export_name = "lumen_rt_apply_2"]
pub fn apply_2(function_boxed_closure: Boxed<Closure>, mut arguments: Vec<Term>) -> Term {
    let arity = function_boxed_closure.arity();
    let arguments_len = arguments.len();

    if arguments_len != (arity as usize) {
        let joined_formatted_elements = arguments
            .iter()
            .map(Term::to_string)
            .collect::<Vec<String>>()
            .join(", ");
        let formatted_arguments = format!("[{}]", joined_formatted_elements);

        panic!(
            "{:?} is arity ({}) does not match arguments ({}) length ({})",
            function_boxed_closure, arity, formatted_arguments, arguments_len
        );
    }

    let callee = function_boxed_closure
        .callee()
        .expect("invalid closure, no code!");

    // If we have a non-empty env, we need to update the argument list
    // to hold the closure value, from which the closure can unpack captured values
    if function_boxed_closure.env_len() > 0 {
        arguments.insert(0, function_boxed_closure.into());
    }

    // We need to prepend the closure value to the callee
    unsafe { erts::apply::apply_callee(callee, arguments.as_slice()) }
}

#[export_name = "lumen_rt_apply_3"]
pub fn apply_3(
    _module_function_arity: ModuleFunctionArity,
    callee: DynamicCallee,
    arguments: Vec<Term>,
) -> Term {
    unsafe { erts::apply::apply_callee(callee, arguments.as_slice()) }
}
