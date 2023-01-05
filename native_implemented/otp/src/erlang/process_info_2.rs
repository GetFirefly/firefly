#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::ptr::NonNull;
use anyhow::*;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::{Atom, atoms, Term};

use crate::runtime::registry::pid_to_process;

#[native_implemented::function(erlang:process_info/2)]
pub fn result(process: &Process, pid: Term, item: Term) -> Result<Term, NonNull<ErlangException>> {
    let pid_pid = term_try_into_local_pid!(pid)?;
    let item_atom: Atom = term_try_into_atom!(item)?;

    if process.pid() == pid_pid {
        process_info(process, item_atom)
    } else {
        match pid_to_process(&pid_pid) {
            Some(pid_arc_process) => process_info(&pid_arc_process, item_atom),
            None => Ok(atoms::Undefined.into()),
        }
    }
    .map_err(From::from)
}

// Private

fn process_info(process: &Process, item: Atom) -> InternalResult<Term> {
    match item.as_str() {
        "backtrace" => unimplemented!(),
        "binary" => unimplemented!(),
        "catchlevel" => unimplemented!(),
        "current_function" => unimplemented!(),
        "current_location" => unimplemented!(),
        "current_stacktrace" => unimplemented!(),
        "dictionary" => unimplemented!(),
        "error_handler" => unimplemented!(),
        "garbage_collection" => unimplemented!(),
        "garbage_collection_info" => unimplemented!(),
        "group_leader" => unimplemented!(),
        "heap_size" => unimplemented!(),
        "initial_call" => unimplemented!(),
        "links" => Ok(links(process)),
        "last_calls" => unimplemented!(),
        "memory" => unimplemented!(),
        "message_queue_len" => unimplemented!(),
        "messages" => Ok(messages(process)),
        "min_heap_size" => unimplemented!(),
        "min_bin_vheap_size" => unimplemented!(),
        "monitored_by" => Ok(monitored_by(process)),
        "monitors" => Ok(monitors(process)),
        "message_queue_data" => unimplemented!(),
        "priority" => unimplemented!(),
        "reductions" => unimplemented!(),
        "registered_name" => Ok(registered_name(process)),
        "sequential_trace_token" => unimplemented!(),
        "stack_size" => unimplemented!(),
        "status" => unimplemented!(),
        "suspending" => unimplemented!(),
        "total_heap_size" => unimplemented!(),
        "trace" => unimplemented!(),
        "trap_exit" => Ok(trap_exit(process)),
        name => Err(TryAtomFromTermError(name))
            .context(
                "supported items are backtrace, binary, catchlevel, current_function, \
                 current_location, current_stacktrace, dictionary, error_handler, \
                 garbage_collection, garbage_collection_info, group_leader, heap_size, \
                 initial_call, links, last_calls, memory, message_queue_len, messages, \
                 min_heap_size, min_bin_vheap_size, monitored_by, monitors, \
                 message_queue_data, priority, reductions, registered_name, \
                 sequential_trace_token, stack_size, status, suspending, \
                 total_heap_size, trace, trap_exit",
            )
            .map_err(From::from),
    }
}

fn links(process: &Process) -> Term {
    let tag = atoms::Links.into();

    let vec: Vec<Term> = process
        .linked_pid_set
        .iter()
        .map(|ref_multi| ref_multi.encode().unwrap())
        .collect();
    let value = process.list_from_slice(&vec).unwrap();

    process.tuple_term_from_term_slice(&[tag, value])
}

fn messages(process: &Process) -> Term {
    let tag = atoms::Messages.into();

    let vec: Vec<Term> = process
        .mailbox
        .lock()
        .borrow()
        .iter()
        .map(|message| match &message.data {
            MessageData::Process(data) => *data,
            MessageData::HeapFragment(message::HeapFragment { data, .. }) => {
                data.clone_to_process(process)
            }
        })
        .collect();

    let value = process.list_from_slice(&vec).unwrap();

    process.tuple_term_from_term_slice(&[tag, value])
}

fn monitored_by(process: &Process) -> Term {
    let tag = atoms::MonitoredBy.into();

    let vec: Vec<Term> = process
        .monitor_by_reference
        .iter()
        .map(|ref_multi| ref_multi.monitoring_pid().encode().unwrap())
        .collect();
    let value = process.list_from_slice(&vec).unwrap();

    process.tuple_term_from_term_slice(&[tag, value])
}

fn monitors(process: &Process) -> Term {
    let monitor_type = atoms::Process.into();
    let mut vec = Vec::new();

    for ref_multi in process.monitored_pid_by_reference.iter() {
        let pid = ref_multi.value();
        let monitor_value = pid.encode().unwrap();
        let monitor = process.tuple_term_from_term_slice(&[monitor_type, monitor_value]);
        vec.push(monitor);
    }

    let tag = atoms::Monitors.into();
    let value = process.list_from_slice(&vec).unwrap();

    process.tuple_term_from_term_slice(&[tag, value])
}

fn registered_name(process: &Process) -> Term {
    match *process.registered_name.read() {
        Some(registered_name) => {
            let tag = atoms::RegisteredName.into();
            let value = registered_name.encode().unwrap();

            process.tuple_term_from_term_slice(&[tag, value])
        }
        None => Term::Nil,
    }
}

fn trap_exit(process: &Process) -> Term {
    let tag = atoms::TrapExit.into();
    let value = process.traps_exit().into();

    process.tuple_term_from_term_slice(&[tag, value])
}
