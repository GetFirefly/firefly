// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;
use std::sync::Arc;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::exception::system::Alloc;
use liblumen_alloc::erts::process::code;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::{badarg, badarity};

use liblumen_alloc::ModuleFunctionArity;

pub fn code(arc_process: &Arc<Process>) -> code::Result {
    arc_process.reduce();

    let function = arc_process.stack_pop().unwrap();
    let arguments = arc_process.stack_pop().unwrap();

    let function_result_boxed_closure: Result<Boxed<Closure>, _> = function.try_into();

    match function_result_boxed_closure {
        Ok(function_boxed_closure) => {
            let mut argument_vec = Vec::new();

            match arguments.decode().unwrap() {
                TypedTerm::Nil => (),
                TypedTerm::List(argument_boxed_cons) => {
                    for result in argument_boxed_cons.into_iter() {
                        match result {
                            Ok(element) => argument_vec.push(element),
                            Err(_) => {
                                arc_process.exception(badarg!());

                                return Ok(());
                            }
                        }
                    }
                }
                _ => {
                    arc_process.exception(badarg!());

                    return Ok(());
                }
            }

            if argument_vec.len() == (function_boxed_closure.arity() as usize) {
                function_boxed_closure.place_frame_with_arguments(
                    arc_process,
                    Placement::Replace,
                    argument_vec,
                )?;

                Process::call_code(arc_process)
            } else {
                match badarity!(arc_process, function, arguments) {
                    exception::Exception::Runtime(runtime_exception) => {
                        arc_process.exception(runtime_exception);

                        Ok(())
                    }
                    exception::Exception::System(system_exception) => Err(system_exception),
                }
            }
        }
        Err(_) => {
            arc_process.exception(badarg!());

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

pub fn module_function_arity() -> Arc<ModuleFunctionArity> {
    Arc::new(ModuleFunctionArity {
        module: module(),
        function: function(),
        arity: 2,
    })
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

// Private

fn frame() -> Frame {
    Frame::new(module_function_arity(), code)
}
