// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;
use std::sync::Arc;

use crate::registry::pid_to_process;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::exception::system::Alloc;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::code::{self, result_from_exception};
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::{atom_unchecked, Atom, Pid, Term};
use liblumen_alloc::{badarg, AsTerm, ModuleFunctionArity};

pub fn place_frame_with_arguments(
    process: &Process,
    placement: Placement,
    pid: Term,
    item: Term,
) -> Result<(), Alloc> {
    process.stack_push(item)?;
    process.stack_push(pid)?;
    process.place_frame(frame(), placement);

    Ok(())
}

// Private

fn code(arc_process: &Arc<Process>) -> code::Result {
    arc_process.reduce();

    let pid = arc_process.stack_pop().unwrap();
    let item = arc_process.stack_pop().unwrap();

    match native(arc_process, pid, item) {
        Ok(info) => {
            arc_process.return_from_call(info)?;

            Process::call_code(arc_process)
        }
        Err(exception) => result_from_exception(arc_process, exception),
    }
}

fn frame() -> Frame {
    Frame::new(module_function_arity(), code)
}

fn function() -> Atom {
    Atom::try_from_str("process_info").unwrap()
}

fn module_function_arity() -> Arc<ModuleFunctionArity> {
    Arc::new(ModuleFunctionArity {
        module: super::module(),
        function: function(),
        arity: 2,
    })
}

fn native(process: &Process, pid: Term, item: Term) -> exception::Result {
    let pid_pid: Pid = pid.try_into()?;
    let item_atom: Atom = item.try_into()?;

    if process.pid() == pid_pid {
        process_info(process, item_atom)
    } else {
        match pid_to_process(&pid_pid) {
            Some(pid_arc_process) => process_info(&pid_arc_process, item_atom),
            None => Ok(atom_unchecked("undefined")),
        }
    }
}

fn process_info(process: &Process, item: Atom) -> exception::Result {
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
        _ => Err(badarg!().into()),
    }
}

fn registered_name(process: &Process) -> exception::Result {
    match *process.registered_name.read() {
        Some(registered_name) => {
            let tag = atom_unchecked("registered_name");
            let value = unsafe { registered_name.as_term() };

            process
                .tuple_from_slice(&[tag, value])
                .map_err(|error| error.into())
        }
        None => Ok(Term::NIL),
    }
}
