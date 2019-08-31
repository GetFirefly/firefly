use core::convert::{TryFrom, TryInto};
use core::result::Result;

use liblumen_alloc::erts::exception::{runtime, Exception};
use liblumen_alloc::term::{Atom, Term, TypedTerm};
use liblumen_alloc::{badarg, Process};

use crate::node;
use crate::registry::{self, pid_to_process};
use crate::scheduler::Scheduler;

pub fn send(
    destination: Term,
    message: Term,
    options: Options,
    process: &Process,
) -> Result<Sent, Exception> {
    match destination.to_typed_term().unwrap() {
        TypedTerm::Atom(destination_atom) => {
            send_to_name(destination_atom, message, options, process)
        }
        TypedTerm::Boxed(unboxed_destination) => {
            match unboxed_destination.to_typed_term().unwrap() {
                TypedTerm::Tuple(tuple) => {
                    if tuple.len() == 2 {
                        let name = tuple[0];

                        match name.to_typed_term().unwrap() {
                            TypedTerm::Atom(name_atom) => {
                                let node = tuple[1];

                                match node.to_typed_term().unwrap() {
                                    TypedTerm::Atom(node_atom) => match node_atom.name() {
                                        node::DEAD => {
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
                _ => Err(badarg!().into()),
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

pub struct Options {
    // Send only suspends for some sends to ports and for remote (`ExternalPid` or
    // `{name, remote_node}`) sends, so it does not apply at this time.
    suspend: bool,
    // Connect only applies when there is distribution, which isn't implemented yet.
    connect: bool,
}

impl Options {
    fn put_option_term(
        &mut self,
        option: Term,
    ) -> core::result::Result<&Options, runtime::Exception> {
        let result: core::result::Result<Atom, _> = option.try_into();

        match result {
            Ok(atom) => match atom.name() {
                "noconnect" => {
                    self.connect = false;

                    Ok(self)
                }
                "nosuspend" => {
                    self.suspend = false;

                    Ok(self)
                }
                _ => Err(badarg!()),
            },
            Err(_) => Err(badarg!()),
        }
    }
}

impl Default for Options {
    fn default() -> Options {
        Options {
            suspend: true,
            connect: true,
        }
    }
}

impl TryFrom<Term> for Options {
    type Error = runtime::Exception;

    fn try_from(term: Term) -> std::result::Result<Options, Self::Error> {
        let mut options: Options = Default::default();
        let mut options_term = term;

        loop {
            match options_term.to_typed_term().unwrap() {
                TypedTerm::Nil => return Ok(options),
                TypedTerm::List(cons) => {
                    options.put_option_term(cons.head)?;
                    options_term = cons.tail;

                    continue;
                }
                _ => return Err(badarg!()),
            }
        }
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
) -> Result<Sent, Exception> {
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
