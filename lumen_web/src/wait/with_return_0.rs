use std::sync::Arc;

use liblumen_alloc::erts::exception::runtime;
use liblumen_alloc::erts::exception::system::Alloc;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::{code, ProcessControlBlock};
use liblumen_alloc::erts::term::{atom_unchecked, Atom, Term};
use liblumen_alloc::erts::ModuleFunctionArity;

use lumen_runtime::scheduler::Scheduler;

pub enum Error {
    NoModuleFunctionArity,
    WrongModuleFunctionArity(Arc<ModuleFunctionArity>),
    NoReturn,
    TooManyReturns(Vec<Term>),
}

/// Put on the stack by the Lumen compiler to allow processes to pause, waiting for the JS event
/// loop to reactivate Lumen and for the returned value placed on the stack to be throw over to JS.
///
/// ```elixir
/// Lumen.Web.Wait.with_return(value)
/// ```
pub fn place_frame_with_arguments(
    process: &ProcessControlBlock,
    placement: Placement,
    value: Term,
) -> Result<(), Alloc> {
    process.stack_push(value)?;
    process.place_frame(frame(), placement);

    Ok(())
}

pub fn pop_return(process: &ProcessControlBlock) -> Result<Term, Error> {
    match process.current_module_function_arity() {
        Some(current_module_function_arity) => {
            if current_module_function_arity.module == super::module()
                && current_module_function_arity.function == function()
                && current_module_function_arity.arity == ARITY
            {
                match process.stack_used() {
                    0 => Err(Error::NoReturn),
                    1 => Ok(process.stack_pop().unwrap()),
                    _ => {
                        let mut stack_vec: Vec<Term> = Vec::new();

                        while let Some(top) = process.stack_pop() {
                            stack_vec.insert(0, top);
                        }

                        Err(Error::TooManyReturns(stack_vec))
                    }
                }
            } else {
                Err(Error::WrongModuleFunctionArity(
                    current_module_function_arity,
                ))
            }
        }
        None => Err(Error::NoModuleFunctionArity),
    }
}

/// Spawns process with this as the first frame, so that any later `Frame`s can return to it.
pub fn spawn(
    parent_process: &ProcessControlBlock,
    heap: *mut Term,
    heap_size: usize,
) -> Result<Arc<ProcessControlBlock>, Alloc> {
    Scheduler::spawn(
        parent_process,
        super::module(),
        function(),
        vec![],
        code,
        heap,
        heap_size,
    )
}

pub fn stop(process: &ProcessControlBlock) {
    process.exit();

    let scheduler_id = process.scheduler_id().unwrap();
    let arc_scheduler = Scheduler::from_id(&scheduler_id).unwrap();
    arc_scheduler.stop_waiting(&process)
}

// Private

const ARITY: u8 = 0;

fn code(arc_process: &Arc<ProcessControlBlock>) -> code::Result {
    let stack_used = arc_process.stack_used();

    match stack_used {
        0 => {
            let reason = atom_unchecked("no_return");
            arc_process.exception(runtime::Exception::exit(
                reason,
                None,
                file!(),
                line!(),
                column!(),
            ))
        }
        1 => Arc::clone(arc_process).wait(),
        _ => {
            let tag = atom_unchecked("too_many_returns");
            let mut stack_vec: Vec<Term> = Vec::new();

            while let Some(top) = arc_process.stack_pop() {
                stack_vec.insert(0, top);
            }

            let stack_list = arc_process.list_from_slice(&stack_vec)?;
            let reason = arc_process.tuple_from_slice(&[tag, stack_list])?;

            arc_process.exception(runtime::Exception::exit(
                reason,
                None,
                file!(),
                line!(),
                column!(),
            ))
        }
    }

    Ok(())
}

fn frame() -> Frame {
    Frame::new(module_function_arity(), code)
}

fn function() -> Atom {
    Atom::try_from_str("with_return").unwrap()
}

fn module_function_arity() -> Arc<ModuleFunctionArity> {
    Arc::new(ModuleFunctionArity {
        module: super::module(),
        function: function(),
        arity: ARITY,
    })
}
