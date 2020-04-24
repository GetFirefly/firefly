// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;

use anyhow::*;

use liblumen_alloc::erts::exception::{self, badarity};
use liblumen_alloc::erts::process::{FrameWithArguments, Process};
use liblumen_alloc::erts::term::prelude::*;

use native_implemented_function::native_implemented_function;

#[native_implemented_function(apply/2)]
fn result(process: &Process, function: Term, arguments: Term) -> exception::Result<Term> {
    let function_result_boxed_closure: Result<Boxed<Closure>, _> = function.try_into();

    match function_result_boxed_closure {
        Ok(function_boxed_closure) => {
            let argument_vec = argument_list_to_vec(arguments)?;

            let arguments_len = argument_vec.len();
            let arity = function_boxed_closure.arity() as usize;

            if arguments_len == arity {
                let frame_with_arguments =
                    function_boxed_closure.frame_with_arguments(false, argument_vec);
                process.queue_frame_with_arguments(frame_with_arguments);

                Ok(Term::NONE)
            } else {
                Err(badarity(
                    process,
                    function,
                    arguments,
                    anyhow!(
                        "arguments ({}) length ({}) does not match arity ({}) of function ({})",
                        arguments,
                        arguments_len,
                        arity,
                        function
                    )
                    .into(),
                ))
            }
        }
        Err(_) => Err(anyhow!(TypeError)
            .context(format!("function ({}) is not a function", function))
            .into()),
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
