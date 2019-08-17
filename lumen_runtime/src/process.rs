///! The memory specific to a process in the VM.
use core::convert::TryInto;

use alloc::sync::Arc;

use hashbrown::HashMap;

use liblumen_core::locks::RwLockWriteGuard;

use liblumen_alloc::erts::exception::runtime;
use liblumen_alloc::erts::exception::system::Alloc;
use liblumen_alloc::erts::process::alloc::heap_alloc::HeapAlloc;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::code::Code;
use liblumen_alloc::erts::process::{self, ProcessControlBlock};
use liblumen_alloc::erts::term::{atom_unchecked, AsTerm, Atom, Term, Tuple, TypedTerm};
use liblumen_alloc::erts::ModuleFunctionArity;
use liblumen_alloc::{CloneToProcess, HeapFragment};

use crate::code;
use crate::otp::erlang::apply_3;
use crate::registry::*;
use crate::scheduler::Scheduler;
use crate::system;
#[cfg(test)]
use crate::test;

fn is_expected_exit_reason(reason: Term) -> bool {
    match reason.to_typed_term().unwrap() {
        TypedTerm::Atom(atom) => match atom.name() {
            "normal" | "shutdown" => true,
            _ => false,
        },
        TypedTerm::Boxed(boxed) => match boxed.to_typed_term().unwrap() {
            TypedTerm::Tuple(tuple) => {
                tuple.len() == 2 && {
                    match tuple[0].to_typed_term().unwrap() {
                        TypedTerm::Atom(atom) => atom.name() == "shutdown",
                        _ => false,
                    }
                }
            }
            _ => false,
        },
        _ => false,
    }
}

pub fn log_exit(process: &ProcessControlBlock, exception: &runtime::Exception) {
    match exception.class {
        runtime::Class::Exit => {
            let reason = exception.reason;

            if !is_expected_exit_reason(reason) {
                system::io::puts(&format!(
                    "** (EXIT from {}) exited with reason: {}",
                    process, reason
                ));
            }
        }
        runtime::Class::Error { .. } => system::io::puts(&format!(
            "** (EXIT from {}) exited with reason: an exception was raised: {}\n{}",
            process,
            exception.reason,
            process.stacktrace()
        )),
        _ => unimplemented!("{:?}", exception),
    }
}

pub fn propagate_exit(process: &ProcessControlBlock, exception: &runtime::Exception) {
    let tag = atom_unchecked("EXIT");
    let from = process.pid_term();
    let reason = exception.reason;
    let exit_message_elements: &[Term] = &[tag, from, reason];
    let exit_message_word_size = Tuple::need_in_words_from_elements(exit_message_elements);

    for linked_pid in process.linked_pid_set.lock().iter() {
        if let Some(linked_pid_arc_process) = pid_to_process(linked_pid) {
            if linked_pid_arc_process.traps_exit() {
                match linked_pid_arc_process.try_acquire_heap() {
                    Some(ref mut linked_pid_heap) => {
                        if exit_message_word_size <= linked_pid_heap.heap_available() {
                            let linked_pid_data = linked_pid_heap
                                .tuple_from_slice(exit_message_elements)
                                .unwrap();

                            linked_pid_arc_process.send_from_self(linked_pid_data);
                        } else {
                            let (heap_fragment_data, heap_fragment) =
                                HeapFragment::tuple_from_slice(exit_message_elements).unwrap();

                            linked_pid_arc_process
                                .send_heap_message(heap_fragment, heap_fragment_data);
                        }
                    }
                    None => {
                        let (heap_fragment_data, heap_fragment) =
                            HeapFragment::tuple_from_slice(exit_message_elements).unwrap();

                        linked_pid_arc_process.send_heap_message(heap_fragment, heap_fragment_data);
                    }
                }
            } else {
                // only tell the linked process to exit.  When it is run by its scheduler, it will
                // go through propagating its own exit.
                linked_pid_arc_process.exit();
            }
        }
    }
}

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

pub fn init(minimum_heap_size: usize) -> Result<ProcessControlBlock, Alloc> {
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

pub trait SchedulerDependentAlloc {
    fn next_reference(&self) -> Result<Term, Alloc>;
}

impl SchedulerDependentAlloc for ProcessControlBlock {
    fn next_reference(&self) -> Result<Term, Alloc> {
        let scheduler_id = self.scheduler_id().unwrap();
        let arc_scheduler = Scheduler::from_id(&scheduler_id).unwrap();
        let number = arc_scheduler.next_reference_number();

        self.reference_from_scheduler(scheduler_id, number)
    }
}

/// Spawns a process with `arguments` on its stack and `code` run with those arguments instead
/// of passing through `apply/3`.
pub fn spawn(
    parent_process: &ProcessControlBlock,
    module: Atom,
    function: Atom,
    arguments: Vec<Term>,
    code: Code,
    heap: *mut Term,
    heap_size: usize,
) -> Result<ProcessControlBlock, Alloc> {
    let arity = arguments.len() as u8;
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

    for argument in arguments.iter().rev() {
        let process_argument = argument.clone_to_process(&process_control_block);
        process_control_block.stack_push(process_argument)?;
    }

    let frame = Frame::new(module_function_arity, code);
    process_control_block.push_frame(frame);

    Ok(process_control_block)
}

/// Spawns a process with arguments for `apply(module, function, arguments)` on its stack.
///
/// This allows the `apply/3` code to be changed with `apply_3::set_code(code)` to handle new
/// MFA unique to a given application.
pub fn spawn_apply_3(
    parent_process: &ProcessControlBlock,
    module: Atom,
    function: Atom,
    arguments: Term,
    heap: *mut Term,
    heap_size: usize,
) -> Result<ProcessControlBlock, Alloc> {
    let arity: u8 = match arguments.to_typed_term().unwrap() {
        TypedTerm::Nil => 0,
        TypedTerm::List(cons) => cons.count().unwrap().try_into().unwrap(),
        _ => {
            panic!(
                "Arguments {:?} are neither an empty nor a proper list",
                arguments
            );
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

    let module_term = unsafe { module.as_term() };
    let function_term = unsafe { function.as_term() };
    let heap_arguments = arguments.clone_to_process(&process_control_block);

    apply_3::place_frame_with_arguments(
        &process_control_block,
        Placement::Push,
        module_term,
        function_term,
        heap_arguments,
    )?;

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
    let module = test::r#loop::module();
    let function = test::r#loop::function();
    let arguments = vec![];
    let code = test::r#loop::code;

    Scheduler::spawn(
        parent_process,
        module,
        function,
        arguments,
        code,
        heap,
        heap_size,
    )
    .unwrap()
}
