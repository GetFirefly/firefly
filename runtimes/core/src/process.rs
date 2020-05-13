pub mod monitor;
pub mod spawn;

use std::cell::RefCell;
use std::convert::TryInto;
use std::sync::Arc;

use liblumen_alloc::erts::exception::{self, AllocResult, ArcError, RuntimeException};
use liblumen_alloc::erts::process::alloc::{Heap, TermAlloc};
use liblumen_alloc::erts::process::{Process, ProcessHeap};
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::{atom, CloneToProcess, HeapFragment, Monitor};

use crate::registry::*;
use crate::scheduler::SchedulerDependentAlloc;
use crate::sys;

thread_local! {
  pub static CURRENT_PROCESS: RefCell<Option<Arc<Process>>> = RefCell::new(None);
}

pub fn current_process() -> Arc<Process> {
    CURRENT_PROCESS.with(|cp| cp.borrow().clone().expect("no process currently scheduled"))
}

pub fn maybe_current_process() -> Option<Arc<Process>> {
    CURRENT_PROCESS.with(|cp| cp.borrow().clone())
}

pub fn is_expected_exception(exception: &RuntimeException) -> bool {
    use exception::Class;
    match exception.class() {
        Some(Class::Exit) => is_expected_exit_reason(exception.reason().unwrap()),
        _ => false,
    }
}

fn is_expected_exit_reason(reason: Term) -> bool {
    match reason.decode().unwrap() {
        TypedTerm::Atom(atom) => match atom.name() {
            "normal" | "shutdown" => true,
            _ => false,
        },
        TypedTerm::Tuple(tuple) => {
            tuple.len() == 2 && {
                match tuple[0].decode().unwrap() {
                    TypedTerm::Atom(atom) => atom.name() == "shutdown",
                    _ => false,
                }
            }
        }
        _ => false,
    }
}

pub fn log_exit(process: &Process, exception: &RuntimeException) {
    use exception::Class;
    match exception.class() {
        Some(Class::Exit) => {
            let reason = exception.reason().unwrap();

            if !is_expected_exit_reason(reason) {
                sys::io::puts(&format!(
                    "** (EXIT from {}) exited with reason: {}",
                    process, reason
                ));
            }
        }
        Some(Class::Error { .. }) => sys::io::puts(&format!(
            "** (EXIT from {}) exited with reason: an exception was raised: {}\n{}",
            process,
            exception.reason().unwrap(),
            process.stacktrace()
        )),
        _ => unimplemented!("{:?}", exception),
    }
}

pub fn monitor(process: &Process, monitored_process: &Process) -> AllocResult<Term> {
    let reference = process.next_reference()?;

    let reference_reference: Boxed<Reference> = reference.try_into().unwrap();
    let monitor = Monitor::Pid {
        monitoring_pid: process.pid(),
    };
    process.monitor(
        reference_reference.as_ref().clone(),
        monitored_process.pid(),
    );
    monitored_process.monitored(reference_reference.as_ref().clone(), monitor);

    Ok(reference)
}

pub fn propagate_exit(process: &Process, exception: &RuntimeException) {
    monitor::propagate_exit(process, exception);
    propagate_exit_to_links(process, exception);
}

pub fn propagate_exit_to_links(process: &Process, exception: &RuntimeException) {
    if !is_expected_exception(exception) {
        let tag = atom!("EXIT");
        let from = process.pid_term();
        let reason = exception.reason().unwrap_or_else(|| atom!("system_error"));
        let reason_word_size = reason.size_in_words();
        let exit_message_elements: &[Term] = &[tag, from, reason];
        let exit_message_word_size = Tuple::need_in_words_from_elements(exit_message_elements);
        let source: ArcError = exception
            .source()
            .context(format!("propagating exit from {}", process));

        for linked_pid in process.linked_pid_set.iter() {
            if let Some(linked_pid_arc_process) = pid_to_process(linked_pid.key()) {
                if linked_pid_arc_process.traps_exit() {
                    match linked_pid_arc_process.try_acquire_heap() {
                        Some(ref mut linked_pid_heap) => {
                            if exit_message_word_size <= linked_pid_heap.heap_available() {
                                send_self_exit_message(
                                    &linked_pid_arc_process,
                                    linked_pid_heap,
                                    exit_message_elements,
                                );
                            } else {
                                send_heap_exit_message(
                                    &linked_pid_arc_process,
                                    exit_message_elements,
                                );
                            }
                        }
                        None => {
                            send_heap_exit_message(&linked_pid_arc_process, exit_message_elements);
                        }
                    }
                } else {
                    // only tell the linked process to exit.  When it is run by its scheduler, it
                    // will go through propagating its own exit.
                    match linked_pid_arc_process.try_acquire_heap() {
                        Some(ref mut linked_pid_heap) => {
                            if reason_word_size <= linked_pid_heap.heap_available() {
                                exit_in_heap(
                                    &linked_pid_arc_process,
                                    linked_pid_heap,
                                    reason,
                                    source.clone(),
                                );
                            } else {
                                exit_in_heap_fragment(
                                    &linked_pid_arc_process,
                                    reason,
                                    source.clone(),
                                );
                            }
                        }
                        None => {
                            exit_in_heap_fragment(&linked_pid_arc_process, reason, source.clone());
                        }
                    }
                }
            }
        }
    }
}

fn send_self_exit_message(
    process: &Process,
    heap: &mut ProcessHeap,
    exit_message_elements: &[Term],
) {
    let data = heap
        .tuple_from_slice(exit_message_elements)
        .unwrap()
        .encode()
        .unwrap();

    process.send_from_self(data);
}

fn send_heap_exit_message(process: &Process, exit_message_elements: &[Term]) {
    let (layout, _) = Tuple::layout_for(exit_message_elements);
    let mut heap_fragment = HeapFragment::new(layout).unwrap();
    let heap_fragment_ref = unsafe { heap_fragment.as_mut() };

    let ptr = heap_fragment_ref
        .tuple_from_slice(exit_message_elements)
        .unwrap();
    process.send_heap_message(heap_fragment, ptr.into());
}

fn exit_in_heap(process: &Process, heap: &mut ProcessHeap, reason: Term, source: ArcError) {
    let data = reason.clone_to_heap(heap).unwrap();

    process.exit(data, source);
}

fn exit_in_heap_fragment(process: &Process, reason: Term, source: ArcError) {
    let (heap_fragment_data, mut heap_fragment) = reason.clone_to_fragment().unwrap();

    process.attach_fragment(unsafe { heap_fragment.as_mut() });
    process.exit(heap_fragment_data, source);
}
