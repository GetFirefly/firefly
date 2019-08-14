// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;
use std::sync::Arc;

use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::exception::system::Alloc;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::code::{self, result_from_exception};
use liblumen_alloc::erts::process::{default_heap, ProcessControlBlock};
use liblumen_alloc::erts::term::{AsTerm, Atom, Term, TypedTerm};
use liblumen_alloc::ModuleFunctionArity;

use crate::scheduler::Scheduler;

pub fn native(
    process_control_block: &ProcessControlBlock,
    module: Term,
    function: Term,
    arguments: Term,
) -> exception::Result {
    let module_atom: Atom = module.try_into()?;
    let function_atom: Atom = function.try_into()?;

    let option_pid = match arguments.to_typed_term().unwrap() {
        TypedTerm::Nil => {
            let (heap, heap_size) = default_heap()?;
            let arc_process = Scheduler::spawn_apply_3(
                process_control_block,
                module_atom,
                function_atom,
                arguments,
                heap,
                heap_size,
            )?;

            Some(arc_process.pid())
        }
        TypedTerm::List(cons) => {
            if cons.is_proper() {
                let (heap, heap_size) = default_heap()?;
                let arc_process = Scheduler::spawn_apply_3(
                    process_control_block,
                    module_atom,
                    function_atom,
                    arguments,
                    heap,
                    heap_size,
                )?;

                Some(arc_process.pid())
            } else {
                None
            }
        }
        _ => None,
    };

    match option_pid {
        Some(pid) => Ok(unsafe { pid.as_term() }),
        None => Err(badarg!().into()),
    }
}

pub fn place_frame_with_arguments(
    process: &ProcessControlBlock,
    placement: Placement,
    module: Term,
    function: Term,
    arguments: Term,
) -> Result<(), Alloc> {
    process.stack_push(arguments)?;
    process.stack_push(function)?;
    process.stack_push(module)?;
    process.place_frame(frame(), placement);

    Ok(())
}

// Private

fn code(arc_process: &Arc<ProcessControlBlock>) -> code::Result {
    arc_process.reduce();

    let module = arc_process.stack_pop().unwrap();
    let function = arc_process.stack_pop().unwrap();
    let arguments = arc_process.stack_pop().unwrap();

    match native(arc_process, module, function, arguments) {
        Ok(child_pid) => {
            arc_process.return_from_call(child_pid)?;

            ProcessControlBlock::call_code(arc_process)
        }
        Err(exception) => result_from_exception(arc_process, exception),
    }
}

fn frame() -> Frame {
    Frame::new(module_function_arity(), code)
}

fn function() -> Atom {
    Atom::try_from_str("spawn").unwrap()
}

fn module_function_arity() -> Arc<ModuleFunctionArity> {
    Arc::new(ModuleFunctionArity {
        module: super::module(),
        function: function(),
        arity: 3,
    })
}
