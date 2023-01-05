use core::convert::TryFrom;

use anyhow::*;

use firefly_rt::term::{Atom, Term, Tuple};
use crate::proplist::TryPropListFromTermError;

use crate::runtime::context::*;
use crate::runtime::proplist::*;

pub struct Options {
    pub reference_frame: ReferenceFrame,
}

const SUPPORTED_OPTIONS_CONTEXT: &str = "supported option is {:abs, bool}";

impl Options {
    fn put_option_term(&mut self, option: Term) -> Result<&Options, anyhow::Error> {
        let tuple: NonNull<Tuple> = option.try_into().context(SUPPORTED_OPTIONS_CONTEXT)?;

        if tuple.len() == 2 {
            let atom: Atom = tuple[0]
                .try_into()
                .map_err(|_| TryPropListFromTermError::KeywordKeyType)
                .context(SUPPORTED_OPTIONS_CONTEXT)?;

            match atom.as_str() {
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
                    .context(SUPPORTED_OPTIONS_CONTEXT),
            }
        } else {
            Err(TryPropListFromTermError::TupleNotPair).context(SUPPORTED_OPTIONS_CONTEXT)
        }
    }
}

impl Default for Options {
    fn default() -> Options {
        Options {
            reference_frame: ReferenceFrame::Relative,
        }
    }
}

impl TryFrom<Term> for Options {
    type Error = anyhow::Error;

    fn try_from(term: Term) -> Result<Self, Self::Error> {
        let mut options: Options = Default::default();
        let mut options_term = term;

        loop {
            match options_term {
                Term::Nil => return Ok(options),
                Term::Cons(non_null_cons) => {
                    let cons = unsafe { non_null_cons.as_ref() };
                    options.put_option_term(cons.head())?;
                    options_term = cons.tail();

                    continue;
                }
                _ => return Err(ImproperListError.into()),
            }
        }
    }
}

pub enum ReferenceFrame {
    Relative,
    Absolute,
}
