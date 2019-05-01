use std::convert::{TryFrom, TryInto};

use crate::exception::Exception;
use crate::list::Cons;
use crate::term::{Tag::*, Term};
use crate::tuple::Tuple;

pub struct Options {
    pub reference_frame: ReferenceFrame,
}

impl Options {
    fn put_option_term(&mut self, option: Term) -> Result<&Options, Exception> {
        match option.tag() {
            Boxed => {
                let unboxed_option: &Term = option.unbox_reference();

                match unboxed_option.tag() {
                    Arity => {
                        let tuple: &Tuple = option.unbox_reference();

                        if tuple.len() == 2 {
                            let name = tuple[0];

                            match name.tag() {
                                Atom => match unsafe { name.atom_to_string() }.as_ref().as_ref() {
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
                                },
                                _ => Err(badarg!()),
                            }
                        } else {
                            Err(badarg!())
                        }
                    }
                    _ => Err(badarg!()),
                }
            }
            _ => Err(badarg!()),
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
            match options_term.tag() {
                EmptyList => return Ok(options),
                List => {
                    let cons: &Cons = unsafe { options_term.as_ref_cons_unchecked() };

                    options.put_option_term(cons.head())?;
                    options_term = cons.tail();

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
