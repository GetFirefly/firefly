pub mod spawn;

use alloc::sync::Arc;

use hashbrown::HashMap;

use liblumen_core::locks::RwLockWriteGuard;

use liblumen_alloc::erts::exception::runtime;
use liblumen_alloc::erts::exception::system::Alloc;
use liblumen_alloc::erts::process::alloc::heap_alloc::HeapAlloc;
use liblumen_alloc::erts::process::code::stack::frame::Frame;
use liblumen_alloc::erts::process::{self, ProcessControlBlock};
use liblumen_alloc::erts::term::{atom_unchecked, Atom, Pid, Reference, Term, Tuple, TypedTerm};
use liblumen_alloc::erts::ModuleFunctionArity;
use liblumen_alloc::{AsTerm, CloneToProcess, HeapFragment, Monitor};

use crate::code;
use crate::otp::erlang::node_0;
#[cfg(test)]
use crate::process::spawn::options::Options;
use crate::registry::*;
use crate::scheduler::Scheduler;
use crate::system;
#[cfg(test)]
use crate::test;

fn is_expected_exception(exception: &runtime::Exception) -> bool {
    match exception.class {
        runtime::Class::Exit => is_expected_exit_reason(exception.reason),
        _ => false,
    }
}

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
    propagate_exit_to_monitors(process, exception);
    propagate_exit_to_links(process, exception);
}

pub fn propagate_exit_to_links(process: &ProcessControlBlock, exception: &runtime::Exception) {
    if !is_expected_exception(exception) {
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

                            linked_pid_arc_process
                                .send_heap_message(heap_fragment, heap_fragment_data);
                        }
                    }
                } else {
                    // only tell the linked process to exit.  When it is run by its scheduler, it
                    // will go through propagating its own exit.
                    linked_pid_arc_process.exit();
                }
            }
        }
    }
}

pub fn propagate_exit_to_monitors(process: &ProcessControlBlock, exception: &runtime::Exception) {
    let info = exception.reason;

    for (reference, monitor) in process.monitor_by_reference.lock().iter() {
        if let Some(monitoring_pid_arc_process) = pid_to_process(&monitor.monitoring_pid()) {
            let down_message_need_in_words = down_need_in_words(monitor, info);

            match monitoring_pid_arc_process.try_acquire_heap() {
                Some(ref mut monitoring_heap) => {
                    if down_message_need_in_words <= monitoring_heap.heap_available() {
                        let monitoring_heap_data =
                            down(monitoring_heap, reference, process, monitor, info);

                        monitoring_pid_arc_process.send_from_self(monitoring_heap_data);
                    } else {
                        send_heap_down_message(
                            &monitoring_pid_arc_process,
                            down_message_need_in_words,
                            reference,
                            process,
                            monitor,
                            info,
                        );
                    }
                }
                None => {
                    send_heap_down_message(
                        &monitoring_pid_arc_process,
                        down_message_need_in_words,
                        reference,
                        process,
                        monitor,
                        info,
                    );
                }
            }
        }
    }
}

pub fn identifier<A: HeapAlloc>(
    process: &ProcessControlBlock,
    monitor: &Monitor,
    heap: &mut A,
) -> Term {
    match monitor {
        Monitor::Pid { .. } => process.pid_term(),
        Monitor::Name { monitored_name, .. } => {
            let monitored_name_term = unsafe { monitored_name.as_term() };
            let node_name = node_0();

            heap.tuple_from_slice(&[monitored_name_term, node_name])
                .unwrap()
        }
    }
}

pub fn identifier_need_in_words(monitor: &Monitor) -> usize {
    match monitor {
        Monitor::Pid { .. } => Pid::SIZE_IN_WORDS,
        Monitor::Name { .. } => {
            Tuple::need_in_words_from_len(2) + Atom::SIZE_IN_WORDS + Atom::SIZE_IN_WORDS
        }
    }
}

pub fn down<A: HeapAlloc>(
    heap: &mut A,
    reference: &Reference,
    process: &ProcessControlBlock,
    monitor: &Monitor,
    info: Term,
) -> Term {
    let tag = atom_unchecked("DOWN");
    let reference_term = reference.clone_to_heap(heap).unwrap();
    let r#type = atom_unchecked("process");
    let identifier = identifier(process, monitor, heap);
    let heap_info = info.clone_to_heap(heap).unwrap();

    heap.tuple_from_slice(&[tag, reference_term, r#type, identifier, heap_info])
        .unwrap()
}

pub fn down_need_in_words(monitor: &Monitor, info: Term) -> usize {
    let identifier_need_in_words = identifier_need_in_words(monitor);

    Tuple::need_in_words_from_len(5)
        + Atom::SIZE_IN_WORDS
        + Reference::need_in_words()
        + Atom::SIZE_IN_WORDS
        + identifier_need_in_words
        + info.size_in_words()
}

pub fn send_heap_down_message(
    monitoring_process: &ProcessControlBlock,
    down_message_need_in_words: usize,
    reference: &Reference,
    monitored_process: &ProcessControlBlock,
    monitor: &Monitor,
    info: Term,
) {
    let mut non_null_heap_fragment =
        unsafe { HeapFragment::new_from_word_size(down_message_need_in_words).unwrap() };
    let heap_fragment = unsafe { non_null_heap_fragment.as_mut() };

    let heap_fragment_data = down(heap_fragment, reference, monitored_process, monitor, info);

    monitoring_process.send_heap_message(non_null_heap_fragment, heap_fragment_data);
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
    let mut options: Options = Default::default();
    options.min_heap_size = Some(16_000);
    let module = test::r#loop::module();
    let function = test::r#loop::function();
    let arguments = vec![];
    let code = test::r#loop::code;

    Scheduler::spawn_code(parent_process, options, module, function, arguments, code).unwrap()
}
