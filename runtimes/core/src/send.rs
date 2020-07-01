mod options;

use std::convert::TryInto;

use anyhow::*;

use liblumen_alloc::erts::exception::InternalResult;
use liblumen_alloc::term::prelude::*;
use liblumen_alloc::Process;

use crate::distribution::nodes::node;
use crate::registry::{self, pid_to_process};
use crate::scheduler;

pub use options::*;

pub fn send(
    destination: Term,
    message: Term,
    options: Options,
    process: &Process,
) -> InternalResult<Sent> {
    match destination.decode()? {
        TypedTerm::Atom(destination_atom) => {
            send_to_name(destination_atom, message, options, process)
        }
        TypedTerm::Tuple(tuple_box) => {
            if tuple_box.len() == 2 {
                let name = tuple_box[0];
                let name_atom: Atom = name.try_into().with_context(|| format!("registered_name ({}) in {{registered_name, node}} ({}) destination is not an atom", name, destination))?;

                let node = tuple_box[1];
                let node_atom: Atom = node.try_into().with_context(|| {
                    format!(
                        "node ({}) in {{registered_name, node}} ({}) destination is not an atom",
                        node, destination
                    )
                })?;

                match node_atom.name() {
                    node::DEAD_ATOM_NAME => send_to_name(name_atom, message, options, process),
                    _ => {
                        if !options.connect {
                            Ok(Sent::ConnectRequired)
                        } else if !options.suspend {
                            Ok(Sent::SuspendRequired)
                        } else {
                            unimplemented!("distribution")
                        }
                    }
                }
            } else {
                Err(anyhow!("destination ({}) is a tuple, but not 2-arity", destination).into())
            }
        }
        TypedTerm::Pid(destination_pid) => {
            if destination_pid == process.pid() {
                process.send_from_self(message);

                Ok(Sent::Sent)
            } else {
                match pid_to_process(&destination_pid) {
                    Some(destination_arc_process) => {
                        if destination_arc_process.send_from_other(message)? {
                            let scheduler_id = destination_arc_process.scheduler_id().unwrap();
                            let arc_scheduler = scheduler::from_id(&scheduler_id).unwrap();
                            arc_scheduler.stop_waiting(&destination_arc_process);
                        }

                        Ok(Sent::Sent)
                    }
                    None => Ok(Sent::Sent),
                }
            }
        }
        _ => Err(TypeError)
            .context(format!(
                "destination ({}) is not registered_name (atom), {{registered_name, node}}, or pid",
                destination
            ))
            .map_err(From::from),
    }
}

pub enum Sent {
    Sent,
    SuspendRequired,
    ConnectRequired,
}

// Private

// `options` will only be used once ports are supported
fn send_to_name(
    destination: Atom,
    message: Term,
    _options: Options,
    process: &Process,
) -> InternalResult<Sent> {
    if *process.registered_name.read() == Some(destination) {
        process.send_from_self(message);

        Ok(Sent::Sent)
    } else {
        match registry::atom_to_process(&destination) {
            Some(destination_arc_process) => {
                if destination_arc_process.send_from_other(message)? {
                    let scheduler_id = destination_arc_process.scheduler_id().unwrap();
                    let arc_scheduler = scheduler::from_id(&scheduler_id).unwrap();
                    arc_scheduler.stop_waiting(&destination_arc_process);
                }

                Ok(Sent::Sent)
            }
            None => Err(anyhow!("name ({}) not registered", destination).into()),
        }
    }
}
