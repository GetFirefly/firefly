///! The memory specific to a process in the VM.
use core::alloc::AllocErr;
use core::convert::TryInto;

use alloc::sync::Arc;

use hashbrown::HashMap;

use liblumen_core::locks::RwLockWriteGuard;

use liblumen_alloc::erts::process::code::stack::frame::Frame;
use liblumen_alloc::erts::process::code::Code;
use liblumen_alloc::erts::process::{self, ProcessControlBlock};
#[cfg(test)]
use liblumen_alloc::erts::term::atom_unchecked;
use liblumen_alloc::erts::term::{AsTerm, Atom, Term, TypedTerm};
use liblumen_alloc::erts::ModuleFunctionArity;
use liblumen_alloc::CloneToProcess;

use crate::code;
use crate::registry::*;
use crate::scheduler::Scheduler;

pub fn register_in(
    arc_process_control_block: Arc<ProcessControlBlock>,
    mut writable_registry: RwLockWriteGuard<HashMap<Atom, Registered>>,
    name: Atom,
) -> bool {
    let mut writable_registered_name = arc_process_control_block.registered_name.write();

    match *writable_registered_name {
        None => {
            writable_registry.insert(
                name,
                Registered::Process(Arc::downgrade(&arc_process_control_block)),
            );
            *writable_registered_name = Some(name);

            true
        }
        Some(_) => false,
    }
}

pub fn init(minimum_heap_size: usize) -> Result<ProcessControlBlock, AllocErr> {
    let init = Atom::try_from_str("init").unwrap();
    let module_function_arity = Arc::new(ModuleFunctionArity {
        module: init,
        function: init,
        arity: 0,
    });

    let heap_size = process::next_heap_size(minimum_heap_size);
    let heap = process::heap(heap_size)?;

    let process = ProcessControlBlock::new(
        Default::default(),
        None,
        Arc::clone(&module_function_arity),
        heap,
        heap_size,
    );

    let frame = Frame::new(module_function_arity, code::init);
    process.push_frame(frame);

    Ok(process)
}

pub trait Alloc {
    fn next_reference(&self) -> Result<Term, AllocErr>;
}

impl Alloc for ProcessControlBlock {
    fn next_reference(&self) -> Result<Term, AllocErr> {
        let scheduler_id = self.scheduler_id().unwrap();
        let arc_scheduler = Scheduler::from_id(&scheduler_id).unwrap();
        let number = arc_scheduler.next_reference_number();

        self.reference_from_scheduler(scheduler_id, number)
    }
}

pub fn spawn(
    parent_process: &ProcessControlBlock,
    module: Atom,
    function: Atom,
    arguments: Term,
    code: Code,
    heap: *mut Term,
    heap_size: usize,
) -> Result<ProcessControlBlock, AllocErr> {
    let arity: u8 = match arguments.to_typed_term().unwrap() {
        TypedTerm::Nil => 0,
        TypedTerm::List(cons) => cons.count().unwrap().try_into().unwrap(),
        _ => {
            #[cfg(debug_assertions)]
            panic!(
                "Arguments {:?} are neither an empty nor a proper list",
                arguments
            );
            #[cfg(not(debug_assertions))]
            panic!("Arguments are neither an empty nor a proper list");
        }
    };

    let module_function_arity = Arc::new(ModuleFunctionArity {
        module,
        function,
        arity,
    });

    let process_control_block = ProcessControlBlock::new(
        parent_process.priority,
        Some(parent_process.pid()),
        Arc::clone(&module_function_arity),
        heap,
        heap_size,
    );

    let heap_arguments = arguments.clone_to_process(&process_control_block);
    process_control_block.stack_push(heap_arguments)?;

    let function_term = unsafe { function.as_term() };
    let module_term = unsafe { module.as_term() };

    // Don't need to be cloned into heap because they are atoms
    process_control_block.stack_push(function_term)?;
    process_control_block.stack_push(module_term)?;

    let frame = Frame::new(module_function_arity, code);
    process_control_block.push_frame(frame);

    Ok(process_control_block)
}

#[cfg(test)]
pub fn test_init() -> Arc<ProcessControlBlock> {
    // During test allow multiple unregistered init processes because in tests, the `Scheduler`s
    // keep getting `Drop`ed as threads end.

    Scheduler::current()
        .spawn_init(
            // init process being the parent process needs space for the arguments when spawning
            // child processes.  These will not be GC'd, so it can be a lot of space if proptest
            // needs to generate a lot of processes.
            16_000,
        )
        .unwrap()
}

#[cfg(test)]
pub fn test(parent_process: &ProcessControlBlock) -> Arc<ProcessControlBlock> {
    let heap_size = process::next_heap_size(16_000);
    let heap = process::heap(heap_size).unwrap();
    let erlang = Atom::try_from_str("erlang").unwrap();
    let exit = Atom::try_from_str("exit").unwrap();

    let normal = atom_unchecked("normal");
    let arguments = parent_process.list_from_slice(&[normal]).unwrap();

    Scheduler::spawn(
        parent_process,
        erlang,
        exit,
        arguments,
        code::apply_fn(),
        heap,
        heap_size,
    )
    .unwrap()
}
