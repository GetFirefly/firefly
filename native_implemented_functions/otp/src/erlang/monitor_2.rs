// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;

use anyhow::*;

use liblumen_alloc::atom;
use liblumen_alloc::erts::exception::{self, AllocResult};
use liblumen_alloc::erts::process::{Monitor, Process};
use liblumen_alloc::erts::term::prelude::*;

use native_implemented_function::native_implemented_function;

use crate::erlang::node_0;
use crate::runtime::context::*;
use crate::runtime::scheduler::SchedulerDependentAlloc;
use crate::runtime::{process, registry};

const TYPE_CONTEXT: &str = "supported types are :port, :process, or :time_offset";

#[native_implemented_function(monitor/2)]
pub fn result(process: &Process, r#type: Term, item: Term) -> exception::Result<Term> {
    let type_atom: Atom = r#type.try_into().context(TYPE_CONTEXT)?;

    match type_atom.name() {
        "port" => unimplemented!(),
        "process" => monitor_process_identifier(process, item),
        "time_offset" => unimplemented!(),
        name => Err(TryAtomFromTermError(name))
            .context(TYPE_CONTEXT)
            .map_err(From::from),
    }
}

// Private

fn monitor_process_identifier(
    process: &Process,
    process_identifier: Term,
) -> exception::Result<Term> {
    match process_identifier.decode()? {
        TypedTerm::Atom(atom) => monitor_process_registered_name(process, process_identifier, atom),
        TypedTerm::Pid(pid) => monitor_process_pid(process, process_identifier, pid),
        TypedTerm::ExternalPid(_) => unimplemented!(),
        TypedTerm::Tuple(tuple) => monitor_process_tuple(process, process_identifier, &tuple),
        _ => Err(TypeError)
            .context(PROCESS_IDENTIFIER_CONTEXT)
            .map_err(From::from),
    }
}

fn monitor_process_identifier_noproc(
    process: &Process,
    identifier: Term,
) -> exception::Result<Term> {
    let monitor_reference = process.next_reference()?;
    let noproc_message = noproc_message(process, monitor_reference, identifier)?;
    process.send_from_self(noproc_message);

    Ok(monitor_reference)
}

fn monitor_process_pid(
    process: &Process,
    process_identifier: Term,
    pid: Pid,
) -> exception::Result<Term> {
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
) -> exception::Result<Term> {
    match registry::atom_to_process(&atom) {
        Some(monitored_arc_process) => {
            let reference = process.next_reference()?;

            let reference_reference: Boxed<Reference> = reference.try_into().expect("fail here");
            let monitor = Monitor::Name {
                monitoring_pid: process.pid(),
                monitored_name: atom,
            };
            process.monitor(
                reference_reference.as_ref().clone(),
                monitored_arc_process.pid(),
            );
            monitored_arc_process.monitored(reference_reference.as_ref().clone(), monitor);

            Ok(reference)
        }
        None => {
            let identifier = process.tuple_from_slice(&[process_identifier, node_0::result()])?;

            monitor_process_identifier_noproc(process, identifier)
        }
    }
}

const PROCESS_IDENTIFIER_CONTEXT: &str =
    "process identifier must be `pid | registered_name() | {registered_name(), node()}`";

fn monitor_process_tuple(
    process: &Process,
    _process_identifier: Term,
    tuple: &Tuple,
) -> exception::Result<Term> {
    if tuple.len() == 2 {
        let registered_name = tuple[0];
        let registered_name_atom = term_try_into_atom("registered name", registered_name)?;

        let node = tuple[1];

        if node == node_0::result() {
            monitor_process_registered_name(process, registered_name, registered_name_atom)
        } else {
            let _: Atom = term_try_into_atom!(node)?;

            unimplemented!(
                "node ({:?}) is not the local node ({:?})",
                node,
                node_0::result()
            );
        }
    } else {
        Err(anyhow!(PROCESS_IDENTIFIER_CONTEXT).into())
    }
}

fn noproc_message(process: &Process, reference: Term, identifier: Term) -> AllocResult<Term> {
    let noproc = atom!("noproc");

    down_message(process, reference, identifier, noproc)
}

fn down_message(
    process: &Process,
    reference: Term,
    identifier: Term,
    info: Term,
) -> AllocResult<Term> {
    let down = atom!("DOWN");
    let r#type = atom!("process");

    process.tuple_from_slice(&[down, reference, r#type, identifier, info])
}
