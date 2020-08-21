use liblumen_alloc::erts::exception::{self, AllocResult};
use liblumen_alloc::erts::process::{FrameWithArguments, Native, Process};
use liblumen_alloc::erts::term::prelude::*;

pub use lumen_rt_core::process::{current_process, monitor, replace_log_exit, set_log_exit, spawn};

#[export_name = "lumen_rt_apply_2"]
pub fn apply_2(function_boxed_closure: Boxed<Closure>, arguments: Vec<Term>) -> Term {
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

    let native = unsafe {
        Native::from_ptr(
            function_boxed_closure.native().as_ptr(),
            function_boxed_closure.native_arity(),
        )
    };

    // codegen'd closures extract their environment from themselves, so it is passed as the first
    // argument.
    let function = function_boxed_closure.into();

    match native {
        Native::One(one) => one(function),
        Native::Two(two) => two(function, arguments[0]),
        Native::Three(three) => three(function, arguments[0], arguments[1]),
        Native::Four(four) => four(function, arguments[0], arguments[1], arguments[2]),
        Native::Five(five) => five(
            function,
            arguments[0],
            arguments[1],
            arguments[2],
            arguments[3],
        ),
        _ => unimplemented!("apply/2 for arity ({})", arity),
    }
}

#[export_name = "lumen_rt_process_runnable"]
pub fn runnable<'a>(
    process: &Process,
    _frames_with_arguments_fn: Box<dyn Fn(&Process) -> AllocResult<Vec<FrameWithArguments>> + 'a>,
) -> AllocResult<()> {
    process.runnable(move |_process| Ok(()))
}
