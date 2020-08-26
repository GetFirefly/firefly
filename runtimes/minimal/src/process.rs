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

    let native_arity = function_boxed_closure.native_arity();

    let native =
        unsafe { Native::from_ptr(function_boxed_closure.native().as_ptr(), native_arity) };

    let native_arguments = if function_boxed_closure.env_len() == 0 {
        // without captured environment variables, the closure passing is optimized out and so
        // should not be passed.
        arguments
    } else {
        // codegen'd closures extract their environment from themselves, so it is passed as the
        // first argument.
        let function = function_boxed_closure.into();

        let mut native_arguments = Vec::with_capacity(native_arity as usize);
        native_arguments.push(function);
        native_arguments.append(&mut arguments);

        native_arguments
    };

    apply_3(
        function_boxed_closure.module_function_arity(),
        native,
        native_arguments,
    )
}

#[export_name = "lumen_rt_apply_3"]
pub fn apply_3(
    _module_function_arity: ModuleFunctionArity,
    native: Native,
    arguments: Vec<Term>,
) -> Term {
    native.apply(&arguments)
}
