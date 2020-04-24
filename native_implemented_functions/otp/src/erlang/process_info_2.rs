// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use anyhow::*;

use liblumen_alloc::atom;
use liblumen_alloc::erts::exception::{self, InternalResult};
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use native_implemented_function::native_implemented_function;

use crate::runtime::registry::pid_to_process;

#[native_implemented_function(process_info/2)]
pub fn result(process: &Process, pid: Term, item: Term) -> exception::Result<Term> {
    let pid_pid = term_try_into_local_pid!(pid)?;
    let item_atom: Atom = term_try_into_atom!(item)?;

    if process.pid() == pid_pid {
        process_info(process, item_atom)
    } else {
        match pid_to_process(&pid_pid) {
            Some(pid_arc_process) => process_info(&pid_arc_process, item_atom),
            None => Ok(atom!("undefined")),
        }
    }
    .map_err(From::from)
}

// Private

fn process_info(process: &Process, item: Atom) -> InternalResult<Term> {
    match item.name() {
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
        "links" => unimplemented!(),
        "last_calls" => unimplemented!(),
        "memory" => unimplemented!(),
        "message_queue_len" => unimplemented!(),
        "messages" => unimplemented!(),
        "min_heap_size" => unimplemented!(),
        "min_bin_vheap_size" => unimplemented!(),
        "monitored_by" => unimplemented!(),
        "monitors" => unimplemented!(),
        "message_queue_data" => unimplemented!(),
        "priority" => unimplemented!(),
        "reductions" => unimplemented!(),
        "registered_name" => registered_name(process),
        "sequential_trace_token" => unimplemented!(),
        "stack_size" => unimplemented!(),
        "status" => unimplemented!(),
        "suspending" => unimplemented!(),
        "total_heap_size" => unimplemented!(),
        "trace" => unimplemented!(),
        "trap_exit" => unimplemented!(),
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

fn registered_name(process: &Process) -> InternalResult<Term> {
    match *process.registered_name.read() {
        Some(registered_name) => {
            let tag = atom!("registered_name");
            let value = registered_name.encode()?;

            process
                .tuple_from_slice(&[tag, value])
                .map_err(|error| error.into())
        }
        None => Ok(Term::NIL),
    }
}
