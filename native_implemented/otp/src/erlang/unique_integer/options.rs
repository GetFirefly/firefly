use std::convert::{TryFrom, TryInto};

use anyhow::*;

use liblumen_alloc::erts::term::prelude::*;

use crate::runtime::proplist::*;

pub struct Options {
    pub positive: bool,
    pub monotonic: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            monotonic: false,
            positive: false,
        }
    }
}

const SUPPORTED_OPTION_CONTEXT: &str = "supported options are monotonic or positive";

impl Options {
    fn put_option_term(&mut self, option: Term) -> Result<&Self, anyhow::Error> {
        let atom: Atom = option.try_into().context(SUPPORTED_OPTION_CONTEXT)?;

        match atom.name() {
            "monotonic" => {
                self.monotonic = true;

                Ok(self)
            }
            "positive" => {
                self.positive = true;

                Ok(self)
            }
            name => Err(TryPropListFromTermError::AtomName(name)).context(SUPPORTED_OPTION_CONTEXT),
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
                    options.put_option_term(cons.head)?;
                    options_term = cons.tail;

                    continue;
                }
                _ => return Err(ImproperListError.into()),
            }
        }
    }
}
