// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;

use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::exception::system::Alloc;
use liblumen_alloc::erts::process::{Monitor, Process};
use liblumen_alloc::erts::term::{
    atom_unchecked, Atom, Boxed, Pid, Reference, Term, Tuple, TypedTerm,
};

use lumen_runtime_macros::native_implemented_function;

use crate::otp::erlang::node_0;
use crate::process::{self, SchedulerDependentAlloc};
use crate::registry;

#[native_implemented_function(monitor/2)]
pub fn native(process: &Process, r#type: Term, item: Term) -> exception::Result {
    let type_atom: Atom = r#type.try_into()?;

    match type_atom.name() {
        "port" => unimplemented!(),
        "process" => monitor_process_identifier(process, item),
        "time_offset" => unimplemented!(),
        _ => Err(badarg!().into()),
    }
}

// Private

fn monitor_process_identifier(process: &Process, process_identifier: Term) -> exception::Result {
    match process_identifier.to_typed_term().unwrap() {
        TypedTerm::Atom(atom) => monitor_process_registered_name(process, process_identifier, atom),
        TypedTerm::Pid(pid) => monitor_process_pid(process, process_identifier, pid),
        TypedTerm::Boxed(boxed) => match boxed.to_typed_term().unwrap() {
            TypedTerm::ExternalPid(_) => unimplemented!(),
            TypedTerm::Tuple(tuple) => monitor_process_tuple(process, process_identifier, &tuple),
            _ => Err(badarg!().into()),
        },
        _ => Err(badarg!().into()),
    }
}

fn monitor_process_identifier_noproc(process: &Process, identifier: Term) -> exception::Result {
    let monitor_reference = process.next_reference()?;
    let noproc_message = noproc_message(process, monitor_reference, identifier)?;
    process.send_from_self(noproc_message);

    Ok(monitor_reference)
}

fn monitor_process_pid(process: &Process, process_identifier: Term, pid: Pid) -> exception::Result {
    match registry::pid_to_process(&pid) {
        Some(monitored_arc_process) => {
            process::monitor(process, &monitored_arc_process).map_err(|alloc| alloc.into())
        }
        None => monitor_process_identifier_noproc(process, process_identifier),
    }
}

fn monitor_process_registered_name(
    process: &Process,
    process_identifier: Term,
    atom: Atom,
) -> exception::Result {
    match registry::atom_to_process(&atom) {
        Some(monitored_arc_process) => {
            let reference = process.next_reference()?;

            let reference_reference: Boxed<Reference> = reference.try_into().expect("fail here");
            let monitor = Monitor::Name {
                monitoring_pid: process.pid(),
                monitored_name: atom,
            };
            process.monitor(reference_reference.clone(), monitored_arc_process.pid());
            monitored_arc_process.monitored(reference_reference.clone(), monitor);

            Ok(reference)
        }
        None => {
            let identifier = process.tuple_from_slice(&[process_identifier, node_0::native()])?;

            monitor_process_identifier_noproc(process, identifier)
        }
    }
}

fn monitor_process_tuple(
    process: &Process,
    _process_identifier: Term,
    tuple: &Tuple,
) -> exception::Result {
    if tuple.len() == 2 {
        let registered_name = tuple[0];
        let registered_name_atom: Atom = registered_name.try_into()?;

        let node = tuple[1];

        if node == node_0::native() {
            monitor_process_registered_name(process, registered_name, registered_name_atom)
        } else {
            let _node_atom: Atom = node.try_into()?;

            unimplemented!(
                "node ({:?}) is not the local node ({:?})",
                node,
                node_0::native()
            );
        }
    } else {
        Err(badarg!().into())
    }
}

fn noproc_message(process: &Process, reference: Term, identifier: Term) -> Result<Term, Alloc> {
    let noproc = atom_unchecked("noproc");

    down_message(process, reference, identifier, noproc)
}

fn down_message(
    process: &Process,
    reference: Term,
    identifier: Term,
    info: Term,
) -> Result<Term, Alloc> {
    let down = atom_unchecked("DOWN");
    let r#type = atom_unchecked("process");

    process.tuple_from_slice(&[down, reference, r#type, identifier, info])
}
