#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;
use std::ptr::NonNull;

use anyhow::*;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::{Atom, atoms, Pid, Reference, Term, Tuple};

use crate::erlang::node_0;
use crate::runtime::context::*;
use crate::runtime::scheduler::SchedulerDependentAlloc;
use crate::runtime::{process, registry};

const TYPE_CONTEXT: &str = "supported types are :port, :process, or :time_offset";

#[native_implemented::function(erlang:monitor/2)]
pub fn result(
    process: &Process,
    r#type: Term,
    item: Term,
) -> Result<Term, NonNull<ErlangException>> {
    let type_atom: Atom = r#type.try_into().context(TYPE_CONTEXT)?;

    match type_atom.as_str() {
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
) -> Result<Term, NonNull<ErlangException>> {
    match process_identifier {
        Term::Atom(atom) => Ok(monitor_process_registered_name(
            process,
            process_identifier,
            atom,
        )),
        Term::Pid(pid) => Ok(monitor_process_pid(process, process_identifier, pid)),
        Term::Tuple(tuple) => monitor_process_tuple(process, process_identifier, &tuple),
        _ => Err(TypeError)
            .context(PROCESS_IDENTIFIER_CONTEXT)
            .map_err(From::from),
    }
}

fn monitor_process_identifier_noproc(process: &Process, identifier: Term) -> Term {
    let monitor_reference = process.next_local_reference_term();
    let noproc_message = noproc_message(process, monitor_reference, identifier);
    process.send_from_self(noproc_message);

    monitor_reference
}

fn monitor_process_pid(process: &Process, process_identifier: Term, pid: Pid) -> Term {
    match registry::pid_to_process(&pid) {
        Some(monitored_arc_process) => process::monitor(process, &monitored_arc_process),
        None => monitor_process_identifier_noproc(process, process_identifier),
    }
}

fn monitor_process_registered_name(
    process: &Process,
    process_identifier: Term,
    atom: Atom,
) -> Term {
    match registry::atom_to_process(&atom) {
        Some(monitored_arc_process) => {
            let reference = process.next_local_reference_term();

            let reference_reference: Boxed<Reference> = reference.try_into().unwrap();
            let monitor = Monitor::Name {
                monitoring_pid: process.pid(),
                monitored_name: atom,
            };
            process.monitor(
                reference_reference.as_ref().clone(),
                monitored_arc_process.pid(),
            );
            monitored_arc_process.monitored(reference_reference.as_ref().clone(), monitor);

            reference
        }
        None => {
            let identifier = process.tuple_term_from_term_slice(&[process_identifier, node_0::result()]);

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
) -> Result<Term, NonNull<ErlangException>> {
    if tuple.len() == 2 {
        let registered_name = tuple[0];
        let registered_name_atom = term_try_into_atom("registered name", registered_name)?;

        let node = tuple[1];

        if node == node_0::result() {
            Ok(monitor_process_registered_name(
                process,
                registered_name,
                registered_name_atom,
            ))
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

fn noproc_message(process: &Process, reference: Term, identifier: Term) -> Term {
    let noproc = atoms::Noproc.into();

    down_message(process, reference, identifier, noproc)
}

fn down_message(process: &Process, reference: Term, identifier: Term, info: Term) -> Term {
    let down = atoms::Down.into();
    let r#type = atoms::Process.into();

    process.tuple_term_from_term_slice(&[down, reference, r#type, identifier, info])
}
