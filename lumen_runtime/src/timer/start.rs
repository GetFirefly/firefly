use core::convert::{TryFrom, TryInto};

use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception::runtime::Exception;
use liblumen_alloc::erts::term::prelude::*;

pub struct Options {
    pub reference_frame: ReferenceFrame,
}

impl Options {
    fn put_option_term(&mut self, option: Term) -> Result<&Options, Exception> {
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
                _ => Err(badarg!()),
            }
        } else {
            Err(badarg!())
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
    type Error = Exception;

    fn try_from(term: Term) -> Result<Options, Exception> {
        let mut options: Options = Default::default();
        let mut options_term = term;

        loop {
            match options_term.to_typed_term().unwrap() {
                TypedTerm::Nil => return Ok(options),
                TypedTerm::List(cons) => {
                    options.put_option_term(cons.head)?;
                    options_term = cons.tail;

                    continue;
                }
                _ => return Err(badarg!()),
            }
        }
    }
}

pub enum ReferenceFrame {
    Relative,
    Absolute,
}
