use std::convert::{TryFrom, TryInto};

use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception::runtime;
use liblumen_alloc::erts::term::prelude::*;

// 2-36
pub struct Base(u8);

impl Base {
    pub fn base(&self) -> u8 {
        self.0
    }

    pub fn radix(&self) -> u32 {
        self.0 as u32
    }

    const MIN_BASE: u8 = 2;
    const MAX_BASE: u8 = 36;
}

impl TryFrom<Term> for Base {
    type Error = runtime::Exception;

    fn try_from(term: Term) -> Result<Self, Self::Error> {
        term.to_typed_term().unwrap().try_into()
    }
}

impl TryFrom<TypedTerm> for Base {
    type Error = runtime::Exception;

    fn try_from(typed_term: TypedTerm) -> Result<Self, Self::Error> {
        match typed_term {
            TypedTerm::SmallInteger(small_integer) => {
                let small_integer_isize: isize = small_integer.into();

                if (Self::MIN_BASE as isize) <= small_integer_isize
                    && small_integer_isize <= (Self::MAX_BASE as isize)
                {
                    Ok(Base(small_integer_isize as u8))
                } else {
                    Err(badarg!())
                }
            }
            _ => Err(badarg!()),
        }
    }
}
