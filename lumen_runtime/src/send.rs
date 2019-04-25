use std::convert::TryFrom;
use std::result::Result;

use crate::exception::Exception;
use crate::list::Cons;
use crate::node;
use crate::process::local::pid_to_process;
use crate::process::Process;
use crate::registry::{self, Registered};
use crate::term::{Tag::*, Term};
use crate::tuple::Tuple;

pub fn send(
    destination: Term,
    message: Term,
    options: Options,
    process: &Process,
) -> Result<Sent, Exception> {
    match destination.tag() {
        Atom => send_to_name(destination, message, options, process),
        Boxed => {
            let unboxed_destination: &Term = destination.unbox_reference();

            match unboxed_destination.tag() {
                Arity => {
                    let tuple: &Tuple = destination.unbox_reference();

                    if tuple.len() == 2 {
                        let name = tuple[0];

                        match name.tag() {
                            Atom => {
                                let node = tuple[1];

                                match node.tag() {
                                    Atom => {
                                        match unsafe { node.atom_to_string() }.as_ref().as_ref() {
                                            node::DEAD => {
                                                send_to_name(name, message, options, process)
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
                                        }
                                    }
                                    _ => Err(badarg!()),
                                }
                            }
                            _ => Err(badarg!()),
                        }
                    } else {
                        Err(badarg!())
                    }
                }
                _ => Err(badarg!()),
            }
        }
        LocalPid => {
            if destination.tagged == process.pid.tagged {
                process.send_from_self(message);

                Ok(Sent::Sent)
            } else {
                match pid_to_process(destination) {
                    Some(destination_process_arc) => {
                        destination_process_arc.send_from_other(message);

                        Ok(Sent::Sent)
                    }
                    None => Ok(Sent::Sent),
                }
            }
        }
        _ => Err(badarg!()),
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
    fn put_option_term(&mut self, option: Term) -> std::result::Result<&Options, Exception> {
        match option.tag() {
            Atom => match unsafe { option.atom_to_string() }.as_ref().as_ref() {
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
            _ => Err(badarg!()),
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
    type Error = Exception;

    fn try_from(term: Term) -> std::result::Result<Options, Exception> {
        let mut options: Options = Default::default();
        let mut options_term = term;

        loop {
            match options_term.tag() {
                EmptyList => return Ok(options),
                List => {
                    let cons: &Cons = unsafe { options_term.as_ref_cons_unchecked() };

                    options.put_option_term(cons.head())?;
                    options_term = cons.tail();

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
    destination: Term,
    message: Term,
    _options: Options,
    process: &Process,
) -> Result<Sent, Exception> {
    if *process.registered_name.read().unwrap() == Some(destination) {
        process.send_from_self(message);

        Ok(Sent::Sent)
    } else {
        let readable_registry = registry::RW_LOCK_REGISTERED_BY_NAME.read().unwrap();

        match readable_registry.get(&destination) {
            Some(Registered::Process(destination_process_arc)) => {
                destination_process_arc.send_from_other(message);

                Ok(Sent::Sent)
            }
            None => Err(badarg!()),
        }
    }
}
