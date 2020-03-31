use std::convert::{TryFrom, TryInto};

use anyhow::*;

use liblumen_alloc::erts::term::prelude::*;

use crate::proplist::TryPropListFromTermError;

pub struct Options {
    // Send only suspends for some sends to ports and for remote (`ExternalPid` or
    // `{name, remote_node}`) sends, so it does not apply at this time.
    pub suspend: bool,
    // Connect only applies when there is distribution, which isn't implemented yet.
    pub connect: bool,
}

const SUPPORTED_OPTIONS_CONTEXT: &str = "supported options are noconnect or nosuspend";

impl Options {
    fn put_option_term(&mut self, option: Term) -> core::result::Result<&Options, anyhow::Error> {
        let result: core::result::Result<Atom, _> =
            option.try_into().context(SUPPORTED_OPTIONS_CONTEXT);

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
                name => {
                    Err(TryPropListFromTermError::AtomName(name)).context(SUPPORTED_OPTIONS_CONTEXT)
                }
            },
            Err(_) => {
                Err(TryPropListFromTermError::PropertyType).context(SUPPORTED_OPTIONS_CONTEXT)
            }
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
    type Error = anyhow::Error;

    fn try_from(term: Term) -> std::result::Result<Options, Self::Error> {
        let mut options: Options = Default::default();
        let mut options_term = term;

        loop {
            match options_term.decode().unwrap() {
                TypedTerm::Nil => return Ok(options),
                TypedTerm::List(cons) => {
                    options.put_option_term(cons.head)?;
                    options_term = cons.tail;

                    continue;
                }
                _ => return Err(ImproperListError.into()),
            }
        }
    }
}
