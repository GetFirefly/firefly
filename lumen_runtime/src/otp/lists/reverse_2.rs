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
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::{Atom, Term, TypedTerm};
use liblumen_alloc::{badarg, ModuleFunctionArity};

pub fn place_frame_with_arguments(
    process: &Process,
    placement: Placement,
    list: Term,
    tail: Term,
) -> Result<(), Alloc> {
    process.stack_push(tail)?;
    process.stack_push(list)?;
    process.place_frame(frame(), placement);

    Ok(())
}

// Private

fn code(arc_process: &Arc<Process>) -> code::Result {
    arc_process.reduce();

    let list = arc_process.stack_pop().unwrap();
    let tail = arc_process.stack_pop().unwrap();

    match native(arc_process, list, tail) {
        Ok(reversed_with_tail) => {
            arc_process.return_from_call(reversed_with_tail)?;

            Process::call_code(arc_process)
        }
        Err(exception) => result_from_exception(arc_process, exception),
    }
}

fn frame() -> Frame {
    Frame::new(module_function_arity(), code)
}

fn function() -> Atom {
    Atom::try_from_str("reverse").unwrap()
}

fn module_function_arity() -> Arc<ModuleFunctionArity> {
    Arc::new(ModuleFunctionArity {
        module: super::module(),
        function: function(),
        arity: 2,
    })
}

pub(super) fn native(process: &Process, list: Term, tail: Term) -> exception::Result {
    match list.to_typed_term().unwrap() {
        TypedTerm::Nil => Ok(tail),
        TypedTerm::List(cons) => {
            let mut reversed = tail;

            for result in cons.into_iter() {
                match result {
                    Ok(element) => {
                        reversed = process.cons(element, reversed)?;
                    }
                    Err(_) => return Err(badarg!().into()),
                }
            }

            Ok(reversed)
        }
        _ => Err(badarg!().into()),
    }
}
