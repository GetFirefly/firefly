use std::convert::{TryFrom, TryInto};

use liblumen_alloc::badarg;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::erts::exception::{self, Exception};

pub struct Options {
    pub r#async: bool,
}

impl Options {
    fn put_option_term(&mut self, option: Term) -> exception::Result<&Options> {
        let tuple: Boxed<Tuple> = option.try_into()?;

        if tuple.len() == 2 {
            let atom: Atom = tuple[0].try_into()?;

            match atom.name() {
                "async" => {
                    self.r#async = tuple[1].try_into()?;

                    Ok(self)
                }
                _ => Err(badarg!().into()),
            }
        } else {
            Err(badarg!().into())
        }
    }
}

impl Default for Options {
    fn default() -> Options {
        Options { r#async: false }
    }
}

impl TryFrom<Term> for Options {
    type Error = Exception;

    fn try_from(term: Term) -> Result<Options, Self::Error> {
        let mut options: Options = Default::default();
        let mut options_term = term;

        loop {
            match options_term.decode()? {
                TypedTerm::Nil => return Ok(options),
                TypedTerm::List(cons) => {
                    options.put_option_term(cons.head)?;
                    options_term = cons.tail;

                    continue;
                }
                _ => return Err(badarg!().into()),
            }
        }
    }
}
