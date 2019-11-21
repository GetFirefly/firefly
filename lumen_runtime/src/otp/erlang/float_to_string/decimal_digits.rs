use std::convert::{TryFrom, TryInto};

use liblumen_alloc::erts::term::prelude::*;

// > {decimals, Decimals :: 0..253}
pub struct DecimalDigits(u8);

impl DecimalDigits {
    pub const MAX_U8: u8 = 253;
}

impl Into<usize> for DecimalDigits {
    fn into(self) -> usize {
        self.0 as usize
    }
}

impl TryFrom<Term> for DecimalDigits {
    type Error = TryIntoIntegerError;

    fn try_from(term: Term) -> Result<Self, Self::Error> {
        let decimal_digits_u8: u8 = term.try_into()?;

        if decimal_digits_u8 <= Self::MAX_U8 {
            Ok(Self(decimal_digits_u8))
        } else {
            Err(TryIntoIntegerError::OutOfRange)
        }
    }
}
