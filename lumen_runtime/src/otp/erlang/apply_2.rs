// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;
use std::sync::Arc;

use anyhow::*;

use liblumen_alloc::erts::exception::{badarity, Alloc};
use liblumen_alloc::erts::process::code;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::Arity;

use locate_code::locate_code;

#[locate_code]
pub fn code(arc_process: &Arc<Process>) -> code::Result {
    arc_process.reduce();

    let function = arc_process.stack_peek(1).unwrap();
    let arguments = arc_process.stack_peek(2).unwrap();

    const STACK_USED: usize = 2;

    let function_result_boxed_closure: Result<Boxed<Closure>, _> = function.try_into();

    match function_result_boxed_closure {
        Ok(function_boxed_closure) => {
            let mut argument_vec = Vec::new();

            match arguments.decode()? {
                TypedTerm::Nil => (),
                TypedTerm::List(argument_boxed_cons) => {
                    for result in argument_boxed_cons.into_iter() {
                        match result {
                            Ok(element) => argument_vec.push(element),
                            Err(_) => {
                                arc_process.stack_popn(STACK_USED);
                                arc_process.exception(
                                    anyhow!(ImproperListError)
                                        .context(format!(
                                            "arguments ({}) is not a proper list",
                                            arguments
                                        ))
                                        .into(),
                                );

                                return Ok(());
                            }
                        }
                    }
                }
                _ => {
                    arc_process.stack_popn(STACK_USED);
                    arc_process.exception(
                        anyhow!(TypeError)
                            .context(format!("arguments ({}) is not a list", arguments))
                            .into(),
                    );

                    return Ok(());
                }
            }

            let arguments_len = argument_vec.len();
            let arity = function_boxed_closure.arity() as usize;

            if arguments_len == arity {
                arc_process.stack_popn(STACK_USED);
                function_boxed_closure
                    .place_frame_with_arguments(arc_process, Placement::Replace, argument_vec)
                    .unwrap();

                Process::call_code(arc_process)
            } else {
                let exception = badarity(
                    arc_process,
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
                );
                code::result_from_exception(arc_process, STACK_USED, exception)
            }
        }
        Err(_) => {
            arc_process.stack_popn(STACK_USED);
            arc_process.exception(
                anyhow!(TypeError)
                    .context(format!("function ({}) is not a function", function))
                    .into(),
            );

            Ok(())
        }
    }
}

pub fn function() -> Atom {
    Atom::try_from_str("apply").unwrap()
}

pub fn module() -> Atom {
    super::module()
}

pub fn place_frame_with_arguments(
    process: &Process,
    placement: Placement,
    function: Term,
    arguments: Term,
) -> Result<(), Alloc> {
    process.stack_push(arguments)?;
    process.stack_push(function)?;
    process.place_frame(frame(), placement);

    Ok(())
}

const ARITY: Arity = 2;

// Private

fn frame() -> Frame {
    Frame::new(module(), function(), ARITY, LOCATION, code)
}
