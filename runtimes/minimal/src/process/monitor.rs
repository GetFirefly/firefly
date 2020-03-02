use core::alloc::Layout;
use core::mem;
use std::convert::TryInto;

use liblumen_alloc::erts::exception::RuntimeException;
use liblumen_alloc::erts::process::alloc::{Heap, TermAlloc};
use liblumen_alloc::erts::process::{Monitor, Process};
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::erts::{self, Message};
use liblumen_alloc::{atom, CloneToProcess, HeapFragment};

use lumen_rt_core::registry::pid_to_process;

#[allow(unused)]
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

pub fn propagate_exit(process: &Process, exception: &RuntimeException) {
    let info = exception.reason().unwrap_or_else(|| atom!("system_error"));

    for (reference, monitor) in process.monitor_by_reference.lock().iter() {
        if let Some(monitoring_pid_arc_process) = pid_to_process(&monitor.monitoring_pid()) {
            let down_layout = down_message_layout(monitor, info);
            let down_layout_words = erts::to_word_size(down_layout.size());

            match monitoring_pid_arc_process.try_acquire_heap() {
                Some(ref mut monitoring_heap) => {
                    if down_layout_words <= monitoring_heap.heap_available() {
                        let monitoring_heap_data =
                            down(monitoring_heap, reference, process, monitor, info);

                        monitoring_pid_arc_process.send_from_self(monitoring_heap_data);
                    } else {
                        send_heap_down_message(
                            &monitoring_pid_arc_process,
                            down_layout,
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
                        down_layout,
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

fn down<A: TermAlloc>(
    heap: &mut A,
    reference: &Reference,
    process: &Process,
    monitor: &Monitor,
    info: Term,
) -> Term {
    let tag = down_tag();
    let reference_term = reference.clone_to_heap(heap).unwrap();
    let r#type = Atom::str_to_term("process");
    let identifier = identifier(process, monitor, heap);
    let heap_info = info.clone_to_heap(heap).unwrap();

    heap.tuple_from_slice(&[tag, reference_term, r#type, identifier, heap_info])
        .unwrap()
        .encode()
        .unwrap()
}

fn down_message_layout(monitor: &Monitor, info: Term) -> Layout {
    let id_layout = identifier_layout(monitor);

    let (layout, _) = Tuple::layout_for_len(DOWN_LEN)
        .extend(Layout::new::<Atom>())
        .unwrap();
    let (layout, _) = layout.extend(Reference::layout()).unwrap();
    let (layout, _) = layout.extend(Layout::new::<Atom>()).unwrap();
    let (layout, _) = layout.extend(id_layout).unwrap();

    let info_bytes = info.size_in_words() * mem::size_of::<usize>();
    let info_align = mem::align_of::<usize>();
    let info_layout = Layout::from_size_align(info_bytes, info_align).unwrap();
    let (layout, _) = layout.extend(info_layout).unwrap();

    layout
}

fn down_tag() -> Term {
    Atom::str_to_term("DOWN")
}

fn identifier<A: TermAlloc>(process: &Process, monitor: &Monitor, heap: &mut A) -> Term {
    use crate::distribution::nodes;
    match monitor {
        Monitor::Pid { .. } => process.pid_term(),
        Monitor::Name { monitored_name, .. } => {
            let monitored_name_term = monitored_name.encode().unwrap();
            let node_name = nodes::node::term();

            heap.tuple_from_slice(&[monitored_name_term, node_name])
                .unwrap()
                .encode()
                .unwrap()
        }
    }
}

fn identifier_layout(monitor: &Monitor) -> Layout {
    match monitor {
        Monitor::Pid { .. } => Layout::new::<Pid>(),
        Monitor::Name { .. } => {
            let (atoms, _) = Layout::new::<Atom>().repeat(2).unwrap();
            let (layout, _) = Tuple::layout_for_len(2).extend(atoms).unwrap();
            layout
        }
    }
}

fn send_heap_down_message(
    monitoring_process: &Process,
    down_layout: Layout,
    reference: &Reference,
    monitored_process: &Process,
    monitor: &Monitor,
    info: Term,
) {
    let mut non_null_heap_fragment = HeapFragment::new(down_layout).unwrap();
    let heap_fragment = unsafe { non_null_heap_fragment.as_mut() };

    let heap_fragment_data = down(heap_fragment, reference, monitored_process, monitor, info);

    monitoring_process.send_heap_message(non_null_heap_fragment, heap_fragment_data);
}
