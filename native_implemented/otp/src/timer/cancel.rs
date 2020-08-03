use core::convert::{TryFrom, TryInto};

use anyhow::*;

use liblumen_alloc::erts::term::prelude::*;

use crate::runtime::proplist::TryPropListFromTermError;

pub struct Options {
    pub r#async: bool,
    pub info: bool,
}

const SUPPORTED_OPTIONS_CONTEXT: &str = "supported options are {:async, bool} or {:info, bool}";

impl Options {
    fn put_option_term(&mut self, option: Term) -> Result<&Options, anyhow::Error> {
        let tuple: Boxed<Tuple> = option.try_into().context(SUPPORTED_OPTIONS_CONTEXT)?;

        if tuple.len() == 2 {
            let atom: Atom = tuple[0]
                .try_into()
                .map_err(|_| TryPropListFromTermError::KeywordKeyType)?;

            match atom.name() {
                "async" => {
                    self.r#async = tuple[1].try_into().context("async value must be a bool")?;

                    Ok(self)
                }
                "info" => {
                    self.info = tuple[1].try_into().context("info value must be a bool")?;

                    Ok(self)
                }
                name => Err(TryPropListFromTermError::KeywordKeyName(name))
                    .context(SUPPORTED_OPTIONS_CONTEXT),
            }
        } else {
            Err(TryPropListFromTermError::TupleNotPair.into())
        }
    }
}

impl Default for Options {
    fn default() -> Options {
        Options {
            r#async: false,
            info: true,
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
                        .with_context(|| SUPPORTED_OPTIONS_CONTEXT)?;
                    options_term = cons.tail;

                    continue;
                }
                _ => return Err(ImproperListError).context(SUPPORTED_OPTIONS_CONTEXT),
            }
        }
    }
}
