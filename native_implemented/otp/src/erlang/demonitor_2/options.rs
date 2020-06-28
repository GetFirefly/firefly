use std::convert::{TryFrom, TryInto};

use anyhow::*;

use liblumen_alloc::erts::term::prelude::*;

use crate::runtime::proplist::TryPropListFromTermError;

pub struct Options {
    pub flush: bool,
    pub info: bool,
}

const SUPPORTED_OPTIONS_CONTEXT: &str = "supported options are :flush or :info";

impl Options {
    fn put_option_term(&mut self, term: Term) -> Result<&Self, anyhow::Error> {
        let option_atom: Atom = term
            .try_into()
            .map_err(|_| TryPropListFromTermError::PropertyType)?;

        match option_atom.name() {
            "flush" => {
                self.flush = true;

                Ok(self)
            }
            "info" => {
                self.info = true;

                Ok(self)
            }
            name => Err(TryPropListFromTermError::AtomName(name).into()),
        }
    }
}

impl Default for Options {
    fn default() -> Self {
        Self {
            flush: false,
            info: false,
        }
    }
}

impl TryFrom<Term> for Options {
    type Error = anyhow::Error;

    fn try_from(term: Term) -> Result<Self, Self::Error> {
        let mut options: Options = Default::default();
        let mut options_term = term;

        loop {
            match options_term.decode().unwrap() {
                TypedTerm::Nil => return Ok(options),
                TypedTerm::List(cons) => {
                    options
                        .put_option_term(cons.head)
                        .context(SUPPORTED_OPTIONS_CONTEXT)?;
                    options_term = cons.tail;

                    continue;
                }
                _ => return Err(ImproperListError).context(SUPPORTED_OPTIONS_CONTEXT),
            };
        }
    }
}
