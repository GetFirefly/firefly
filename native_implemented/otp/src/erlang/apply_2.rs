use std::convert::TryInto;

use anyhow::*;

use liblumen_alloc::erts::exception::{self, badarity};
use liblumen_alloc::erts::process::{trace::Trace, FrameWithArguments, Process};
use liblumen_alloc::erts::term::prelude::*;

extern "Rust" {
    #[link_name = "lumen_rt_apply_2"]
    fn runtime_apply_2(function_boxed_closure: Boxed<Closure>, arguments: Vec<Term>) -> Term;
}

#[native_implemented::function(erlang:apply/2)]
fn result(process: &Process, function: Term, arguments: Term) -> exception::Result<Term> {
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
        Err(badarity(
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
        )
        .into())
    }
}

fn argument_list_to_vec(list: Term) -> exception::Result<Vec<Term>> {
    let mut vec = Vec::new();

    match list.decode()? {
        TypedTerm::Nil => Ok(vec),
        TypedTerm::List(boxed_cons) => {
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

pub fn frame_with_arguments(function: Term, arguments: Term) -> FrameWithArguments {
    frame().with_arguments(false, &[function, arguments])
}
