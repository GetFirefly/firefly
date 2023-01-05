use std::ptr::NonNull;

use anyhow::*;

use firefly_rt::backtrace::Trace;
use firefly_rt::error::ErlangException;
use firefly_rt::function::{Arity, ErlangResult, ModuleFunctionArity};
use firefly_rt::process::Process;
use firefly_rt::term::{atoms, Closure, Term, TypeError};

use crate::runtime;

extern "Rust" {
    #[link_name = "lumen_rt_apply_2"]
    fn runtime_apply_2(
        function_boxed_closure: Boxed<Closure>,
        arguments: Vec<Term>,
    ) -> ErlangResult;
}

#[export_name = "erlang:apply/2"]
pub extern "C-unwind" fn apply_2(function: Term, arguments: Term) -> ErlangResult {
    let arc_process = runtime::process::current_process();
    match apply_2_impl(&arc_process, function, arguments) {
        Ok(result) => result,
        Err(exception) => arc_process.return_status(Err(exception)),
    }
}

fn apply_2_impl(
    process: &Process,
    function: Term,
    arguments: Term,
) -> Result<ErlangResult, NonNull<ErlangException>> {
    let function_boxed_closure: Boxed<Closure> = function
        .try_into()
        .with_context(|| format!("function ({}) is not a function", function))?;
    let argument_vec = argument_list_to_vec(arguments)?;
    let arguments_len = argument_vec.len();
    let arity = function_boxed_closure.arity() as usize;

    if arguments_len == arity {
        Ok(unsafe { runtime_apply_2(function_boxed_closure, argument_vec) })
    } else {
        let mfa = function_boxed_closure.module_function_arity();
        let trace = Trace::capture();
        trace.set_top_frame(&mfa, argument_vec.as_slice());
        let exception = badarity(
            process,
            function,
            arguments,
            trace,
            Some(
                anyhow!(
                    "arguments ({}) length ({}) does not match arity ({}) of function ({})",
                    arguments,
                    arguments_len,
                    arity,
                    function
                )
                .into(),
            ),
        );
        Err(Exception::Runtime(exception))
    }
}

fn argument_list_to_vec(list: Term) -> Result<Vec<Term>, NonNull<ErlangException>> {
    let mut vec = Vec::new();

    match list {
        Term::Nil => Ok(vec),
        Term::Cons(boxed_cons) => {
            for result in boxed_cons.into_iter() {
                match result {
                    Ok(element) => vec.push(element),
                    Err(_) => {
                        return Err(anyhow!(ImproperListError)
                            .context(format!("arguments ({}) is not a proper list", list)))
                        .map_err(From::from)
                    }
                }
            }

            Ok(vec)
        }
        _ => Err(anyhow!(TypeError)
            .context(format!("arguments ({}) is not a list", list))
            .into()),
    }
}

pub fn module_function_arity() -> ModuleFunctionArity {
    ModuleFunctionArity {
        module: atoms::Erlang,
        function: atoms::Apply,
        arity: ARITY,
    }
}

pub fn frame_with_arguments(function: Term, arguments: Term) -> FrameWithArguments {
    frame().with_arguments(false, &[function, arguments])
}

pub const ARITY: Arity = 2;
pub const CLOSURE_NATIVE: Option<NonNull<std::ffi::c_void>> =
    Some(unsafe { NonNull::new_unchecked(apply_2 as *mut std::ffi::c_void) });
