mod compression;
mod minor_version;

use std::convert::{TryFrom, TryInto};

use liblumen_alloc::erts::term::prelude::*;

use compression::*;
use minor_version::*;

pub struct Options {
    compression: Compression,
    minor_version: MinorVersion,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            // No compression is done (it is the same as giving no compressed option)
            compression: Compression(0),
            minor_version: Default::default(),
        }
    }
}

impl Options {
    fn put_option_term(&mut self, option: Term) -> Result<&Self, TryFromTermError> {
        match option.decode().unwrap() {
            TypedTerm::Atom(atom) => match atom.name() {
                "compressed" => {
                    self.compression = Default::default();

                    Ok(self)
                }
                name => Err(TryFromTermError::AtomName(name)),
            },
            TypedTerm::Tuple(tuple) => {
                if tuple.len() == 2 {
                    let atom: Atom = tuple[0]
                        .try_into()
                        .map_err(|_| TryFromTermError::KeywordKeyType)?;

                    match atom.name() {
                        "compressed" => {
                            self.compression = tuple[1]
                                .try_into()
                                .map_err(|_| TryFromTermError::CompressedType)?;

                            Ok(self)
                        }
                        "minor_version" => {
                            self.minor_version = tuple[1]
                                .try_into()
                                .map_err(|_| TryFromTermError::MinorVersionType)?;

                            Ok(self)
                        }
                        name => Err(TryFromTermError::KeywordKeyName(name)),
                    }
                } else {
                    Err(TryFromTermError::TupleLen)
                }
            }
            _ => Err(TryFromTermError::ElementType),
        }
    }
}

impl TryFrom<Term> for Options {
    type Error = TryFromTermError;

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
                _ => return Err(TryFromTermError::Type),
            };
        }
    }
}

pub enum TryFromTermError {
    AtomName(&'static str),
    CompressedType,
    MinorVersionType,
    KeywordKeyType,
    KeywordKeyName(&'static str),
    ElementType,
    TupleLen,
    Type,
}
