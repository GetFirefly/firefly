use core::convert::{TryFrom, TryInto};

use anyhow::*;

use liblumen_alloc::erts::term::prelude::*;

use crate::context::term_try_into_bool;
use crate::proplist::TryPropListFromTermError;

#[derive(Clone, Copy, Debug)]
pub enum ReferenceFrame {
    Relative,
    Absolute,
}

pub struct StartOptions {
    pub reference_frame: ReferenceFrame,
}
impl StartOptions {
    const SUPPORTED_OPTIONS_CONTEXT: &'static str = "supported option is {:abs, bool}";

    fn put_option_term(&mut self, option: Term) -> Result<&Self, anyhow::Error> {
        let tuple: Boxed<Tuple> = option.try_into().context(Self::SUPPORTED_OPTIONS_CONTEXT)?;

        if tuple.len() == 2 {
            let atom: Atom = tuple[0]
                .try_into()
                .map_err(|_| TryPropListFromTermError::KeywordKeyType)
                .context(Self::SUPPORTED_OPTIONS_CONTEXT)?;

            match atom.name() {
                "abs" => {
                    let value = tuple[1];
                    let absolute: bool = term_try_into_bool("abs value", value)?;

                    self.reference_frame = if absolute {
                        ReferenceFrame::Absolute
                    } else {
                        ReferenceFrame::Relative
                    };

                    Ok(self)
                }
                name => Err(TryPropListFromTermError::KeywordKeyName(name))
                    .context(Self::SUPPORTED_OPTIONS_CONTEXT),
            }
        } else {
            Err(TryPropListFromTermError::TupleNotPair).context(Self::SUPPORTED_OPTIONS_CONTEXT)
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
    type Error = anyhow::Error;

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
                _ => return Err(ImproperListError.into()),
            }
        }
    }
}

pub struct ReadOptions {
    pub r#async: bool,
}
impl ReadOptions {
    const SUPPORTED_OPTIONS_CONTEXT: &'static str = "supported option is {:async, bool}";

    fn put_option_term(&mut self, option: Term) -> Result<&Self, anyhow::Error> {
        let tuple: Boxed<Tuple> = option.try_into().context(Self::SUPPORTED_OPTIONS_CONTEXT)?;

        if tuple.len() == 2 {
            let atom: Atom = tuple[0]
                .try_into()
                .map_err(|_| TryPropListFromTermError::KeywordKeyType)?;

            match atom.name() {
                "async" => {
                    self.r#async = tuple[1].try_into().context("async value must be a bool")?;

                    Ok(self)
                }
                name => Err(TryPropListFromTermError::KeywordKeyName(name))
                    .context(Self::SUPPORTED_OPTIONS_CONTEXT),
            }
        } else {
            Err(TryPropListFromTermError::TupleNotPair).context(Self::SUPPORTED_OPTIONS_CONTEXT)
        }
    }
}
impl Default for ReadOptions {
    fn default() -> Self {
        Self { r#async: false }
    }
}
impl TryFrom<Term> for ReadOptions {
    type Error = anyhow::Error;

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
                _ => bail!(ImproperListError),
            }
        }
    }
}

pub struct CancellationOptions {
    pub r#async: bool,
    pub info: bool,
}
impl CancellationOptions {
    const SUPPORTED_OPTIONS_CONTEXT: &'static str =
        "supported options are {:async, bool} or {:info, bool}";

    fn put_option_term(&mut self, option: Term) -> Result<&Self, anyhow::Error> {
        let tuple: Boxed<Tuple> = option.try_into().context(Self::SUPPORTED_OPTIONS_CONTEXT)?;

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
                    .context(Self::SUPPORTED_OPTIONS_CONTEXT),
            }
        } else {
            Err(TryPropListFromTermError::TupleNotPair.into())
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
    type Error = anyhow::Error;

    fn try_from(term: Term) -> Result<CancellationOptions, Self::Error> {
        let mut options = Self::default();
        let mut options_term = term;

        loop {
            match options_term.decode()? {
                TypedTerm::Nil => return Ok(options),
                TypedTerm::List(cons) => {
                    options
                        .put_option_term(cons.head)
                        .with_context(|| CancellationOptions::SUPPORTED_OPTIONS_CONTEXT)?;
                    options_term = cons.tail;

                    continue;
                }
                _ => return Err(ImproperListError).context(Self::SUPPORTED_OPTIONS_CONTEXT),
            }
        }
    }
}
