// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::sync::Arc;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::exception::system::Alloc;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::code::{self, result_from_exception};
use liblumen_alloc::erts::process::ProcessControlBlock;
use liblumen_alloc::erts::term::{Atom, Term, TypedTerm};
use liblumen_alloc::{badarg, ModuleFunctionArity};

use crate::registry::pid_to_process;

pub fn place_frame_with_arguments(
    process: &ProcessControlBlock,
    placement: Placement,
    pid_or_port: Term,
) -> Result<(), Alloc> {
    process.stack_push(pid_or_port)?;
    process.place_frame(frame(), placement);

    Ok(())
}

// Private

fn code(arc_process: &Arc<ProcessControlBlock>) -> code::Result {
    arc_process.reduce();

    let pid_or_port = arc_process.stack_pop().unwrap();

    match native(arc_process, pid_or_port) {
        Ok(true_term) => {
            arc_process.return_from_call(true_term)?;

            ProcessControlBlock::call_code(arc_process)
        }
        Err(exception) => result_from_exception(arc_process, exception),
    }
}

fn frame() -> Frame {
    Frame::new(module_function_arity(), code)
}

fn function() -> Atom {
    Atom::try_from_str("unlink").unwrap()
}

fn module_function_arity() -> Arc<ModuleFunctionArity> {
    Arc::new(ModuleFunctionArity {
        module: super::module(),
        function: function(),
        arity: 1,
    })
}

fn native(process: &ProcessControlBlock, pid_or_port: Term) -> exception::Result {
    match pid_or_port.to_typed_term().unwrap() {
        TypedTerm::Pid(pid) => {
            if pid == process.pid() {
                Ok(true.into())
            } else {
                match pid_to_process(&pid) {
                    Some(pid_arc_process) => {
                        process.unlink(&pid_arc_process);
                    }
                    None => (),
                }

                Ok(true.into())
            }
        }
        TypedTerm::Port(_) => unimplemented!(),
        TypedTerm::Boxed(boxed) => match boxed.to_typed_term().unwrap() {
            TypedTerm::ExternalPid(_) => unimplemented!(),
            TypedTerm::ExternalPort(_) => unimplemented!(),
            _ => Err(badarg!().into()),
        },
        _ => Err(badarg!().into()),
    }
}
