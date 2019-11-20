mod options;

use liblumen_alloc::erts::exception;
use liblumen_alloc::term::prelude::*;
use liblumen_alloc::{badarg, Process};

use crate::distribution::nodes::node;
use crate::registry::{self, pid_to_process};
use crate::scheduler::Scheduler;

pub use options::*;

pub fn send(
    destination: Term,
    message: Term,
    options: Options,
    process: &Process,
) -> exception::Result<Sent> {
    match destination.decode().unwrap() {
        TypedTerm::Atom(destination_atom) => {
            send_to_name(destination_atom, message, options, process)
        }
        TypedTerm::Tuple(tuple_box) => {
            if tuple_box.len() == 2 {
                let name = tuple_box[0];

                match name.decode()? {
                    TypedTerm::Atom(name_atom) => {
                        let node = tuple_box[1];

                        match node.decode().unwrap() {
                            TypedTerm::Atom(node_atom) => match node_atom.name() {
                                node::DEAD_ATOM_NAME => {
                                    send_to_name(name_atom, message, options, process)
                                }
                                _ => {
                                    if !options.connect {
                                        Ok(Sent::ConnectRequired)
                                    } else if !options.suspend {
                                        Ok(Sent::SuspendRequired)
                                    } else {
                                        unimplemented!("distribution")
                                    }
                                }
                            },
                            _ => Err(badarg!().into()),
                        }
                    }
                    _ => Err(badarg!().into()),
                }
            } else {
                Err(badarg!().into())
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
                            let arc_scheduler = Scheduler::from_id(&scheduler_id).unwrap();
                            arc_scheduler.stop_waiting(&destination_arc_process);
                        }

                        Ok(Sent::Sent)
                    }
                    None => Ok(Sent::Sent),
                }
            }
        }
        _ => Err(badarg!().into()),
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
) -> exception::Result<Sent> {
    if *process.registered_name.read() == Some(destination) {
        process.send_from_self(message);

        Ok(Sent::Sent)
    } else {
        match registry::atom_to_process(&destination) {
            Some(destination_arc_process) => {
                if destination_arc_process.send_from_other(message)? {
                    let scheduler_id = destination_arc_process.scheduler_id().unwrap();
                    let arc_scheduler = Scheduler::from_id(&scheduler_id).unwrap();
                    arc_scheduler.stop_waiting(&destination_arc_process);
                }

                Ok(Sent::Sent)
            }
            None => Err(badarg!().into()),
        }
    }
}
