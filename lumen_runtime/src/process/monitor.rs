use std::convert::TryInto;

use liblumen_alloc::erts::exception::runtime;
use liblumen_alloc::erts::process::alloc::heap_alloc::HeapAlloc;
use liblumen_alloc::erts::process::{Monitor, Process};
use liblumen_alloc::erts::term::{
    atom_unchecked, AsTerm, Atom, Boxed, Pid, Reference, Term, Tuple,
};
use liblumen_alloc::erts::Message;
use liblumen_alloc::{CloneToProcess, HeapFragment};

use crate::otp::erlang::node_0;
use crate::registry::pid_to_process;

pub fn is_down(message: &Message, reference: &Reference) -> bool {
    let message_data = message.data();

    let result_tuple: Result<Boxed<Tuple>, _> = (*message_data).try_into();

    match result_tuple {
        Ok(tuple) => {
            tuple.len() == DOWN_LEN && {
                let result_message_reference: Result<Boxed<Reference>, _> = tuple[1].try_into();

                match result_message_reference {
                    Ok(message_reference) => &message_reference == reference,
                    Err(_) => false,
                }
            }
        }
        Err(_) => false,
    }
}

pub fn propagate_exit(process: &Process, exception: &runtime::Exception) {
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

// Private

const DOWN_LEN: usize = 5;

fn down<A: HeapAlloc>(
    heap: &mut A,
    reference: &Reference,
    process: &Process,
    monitor: &Monitor,
    info: Term,
) -> Term {
    let tag = down_tag();
    let reference_term = reference.clone_to_heap(heap).unwrap();
    let r#type = atom_unchecked("process");
    let identifier = identifier(process, monitor, heap);
    let heap_info = info.clone_to_heap(heap).unwrap();

    heap.tuple_from_slice(&[tag, reference_term, r#type, identifier, heap_info])
        .unwrap()
}

fn down_need_in_words(monitor: &Monitor, info: Term) -> usize {
    let identifier_need_in_words = identifier_need_in_words(monitor);

    Tuple::need_in_words_from_len(DOWN_LEN)
        + Atom::SIZE_IN_WORDS
        + Reference::need_in_words()
        + Atom::SIZE_IN_WORDS
        + identifier_need_in_words
        + info.size_in_words()
}

fn down_tag() -> Term {
    atom_unchecked("DOWN")
}

fn identifier<A: HeapAlloc>(process: &Process, monitor: &Monitor, heap: &mut A) -> Term {
    match monitor {
        Monitor::Pid { .. } => process.pid_term(),
        Monitor::Name { monitored_name, .. } => {
            let monitored_name_term = unsafe { monitored_name.as_term() };
            let node_name = node_0::native();

            heap.tuple_from_slice(&[monitored_name_term, node_name])
                .unwrap()
        }
    }
}

fn identifier_need_in_words(monitor: &Monitor) -> usize {
    match monitor {
        Monitor::Pid { .. } => Pid::SIZE_IN_WORDS,
        Monitor::Name { .. } => {
            Tuple::need_in_words_from_len(2) + Atom::SIZE_IN_WORDS + Atom::SIZE_IN_WORDS
        }
    }
}

fn send_heap_down_message(
    monitoring_process: &Process,
    down_message_need_in_words: usize,
    reference: &Reference,
    monitored_process: &Process,
    monitor: &Monitor,
    info: Term,
) {
    let mut non_null_heap_fragment =
        unsafe { HeapFragment::new_from_word_size(down_message_need_in_words).unwrap() };
    let heap_fragment = unsafe { non_null_heap_fragment.as_mut() };

    let heap_fragment_data = down(heap_fragment, reference, monitored_process, monitor, info);

    monitoring_process.send_heap_message(non_null_heap_fragment, heap_fragment_data);
}
