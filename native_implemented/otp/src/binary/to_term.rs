use std::convert::{TryFrom, TryInto};

use anyhow::*;

use liblumen_alloc::erts::term::prelude::*;

use crate::runtime::proplist::TryPropListFromTermError;

pub struct Options {
    pub existing: bool,
    pub used: bool,
}

const SUPPORTED_OPTIONS_CONTEXT: &str = "supported options are safe and used";

impl Options {
    fn put_option_term(&mut self, option: Term) -> anyhow::Result<&Options> {
        let atom: Atom = option.try_into().context(SUPPORTED_OPTIONS_CONTEXT)?;

        match atom.name() {
            "safe" => {
                self.existing = true;

                Ok(self)
            }
            "used" => {
                self.used = true;

                Ok(self)
            }
            name => {
                Err(TryPropListFromTermError::AtomName(name)).context(SUPPORTED_OPTIONS_CONTEXT)
            }
        }
    }
}

impl TryFrom<Term> for Options {
    type Error = anyhow::Error;

    fn try_from(term: Term) -> anyhow::Result<Self> {
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
            };
        }
    }
}

impl Default for Options {
    fn default() -> Self {
        Self {
            existing: false,
            used: false,
        }
    }
}
