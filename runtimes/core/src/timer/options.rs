use core::convert::{TryFrom, TryInto};

use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception::{self, Exception};
use liblumen_alloc::erts::term::prelude::*;

#[derive(Clone, Copy, Debug)]
pub enum ReferenceFrame {
    Relative,
    Absolute,
}

pub struct StartOptions {
    pub reference_frame: ReferenceFrame,
}
impl StartOptions {
    fn put_option_term(&mut self, option: Term) -> exception::Result<&Self> {
        let tuple: Boxed<Tuple> = option.try_into()?;

        if tuple.len() == 2 {
            let atom: Atom = tuple[0].try_into()?;

            match atom.name() {
                "abs" => {
                    let absolute: bool = tuple[1].try_into()?;

                    self.reference_frame = if absolute {
                        ReferenceFrame::Absolute
                    } else {
                        ReferenceFrame::Relative
                    };

                    Ok(self)
                }
                _ => Err(badarg!().into()),
            }
        } else {
            Err(badarg!().into())
        }
    }
}
impl Default for StartOptions {
    fn default() -> Self {
        Self {
            reference_frame: ReferenceFrame::Relative,
        }
    }
}
impl TryFrom<Term> for StartOptions {
    type Error = Exception;

    fn try_from(term: Term) -> Result<Self, Self::Error> {
        let mut options = Self::default();
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

pub struct ReadOptions {
    pub r#async: bool,
}
impl ReadOptions {
    fn put_option_term(&mut self, option: Term) -> exception::Result<&Self> {
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
impl Default for ReadOptions {
    fn default() -> Self {
        Self { r#async: false }
    }
}
impl TryFrom<Term> for ReadOptions {
    type Error = Exception;

    fn try_from(term: Term) -> Result<Self, Self::Error> {
        let mut options = Self::default();
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

pub struct CancellationOptions {
    pub r#async: bool,
    pub info: bool,
}
impl CancellationOptions {
    fn put_option_term(&mut self, option: Term) -> exception::Result<&Self> {
        let tuple: Boxed<Tuple> = option.try_into()?;

        if tuple.len() == 2 {
            let atom: Atom = tuple[0].try_into()?;

            match atom.name() {
                "async" => {
                    self.r#async = tuple[1].try_into()?;

                    Ok(self)
                }
                "info" => {
                    self.info = tuple[1].try_into()?;

                    Ok(self)
                }
                _ => Err(badarg!().into()),
            }
        } else {
            Err(badarg!().into())
        }
    }
}
impl Default for CancellationOptions {
    fn default() -> CancellationOptions {
        Self {
            r#async: false,
            info: true,
        }
    }
}
impl TryFrom<Term> for CancellationOptions {
    type Error = Exception;

    fn try_from(term: Term) -> Result<CancellationOptions, Self::Error> {
        let mut options = Self::default();
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
