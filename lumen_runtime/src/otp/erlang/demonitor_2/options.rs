use std::convert::{TryFrom, TryInto};

use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception::Exception;
use liblumen_alloc::erts::term::prelude::*;

pub struct Options {
    pub flush: bool,
    pub info: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            flush: false,
            info: false,
        }
    }
}

impl TryFrom<Boxed<Cons>> for Options {
    type Error = Exception;

    fn try_from(cons: Boxed<Cons>) -> Result<Self, Self::Error> {
        let mut options: Options = Default::default();

        for result in cons.into_iter() {
            match result {
                Ok(option) => {
                    let option_atom: Atom = option.try_into()?;

                    match option_atom.name() {
                        "flush" => {
                            options.flush = true;
                        }
                        "info" => {
                            options.info = true;
                        }
                        _ => return Err(badarg!().into()),
                    }
                }
                Err(_) => return Err(badarg!().into()),
            }
        }

        Ok(options)
    }
}

impl TryFrom<Term> for Options {
    type Error = Exception;

    fn try_from(term: Term) -> Result<Self, Self::Error> {
        term.decode().unwrap().try_into()
    }
}

impl TryFrom<TypedTerm> for Options {
    type Error = Exception;

    fn try_from(typed_term: TypedTerm) -> Result<Self, Self::Error> {
        match typed_term {
            TypedTerm::Nil => Ok(Default::default()),
            TypedTerm::List(cons) => cons.try_into(),
            _ => Err(badarg!().into()),
        }
    }
}
