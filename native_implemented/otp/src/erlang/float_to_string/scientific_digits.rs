use std::convert::{TryFrom, TryInto};

use firefly_number::TryIntoIntegerError;
use firefly_rt::term::Term;

pub struct ScientificDigits(u8);

impl ScientificDigits {
    // > {scientific, Decimals :: 0..249}
    pub const MAX_U8: u8 = 249;
}

impl Default for ScientificDigits {
    fn default() -> Self {
        // > [float_binary(float) is the] same as float_to_binary(Float,[{scientific,20}]).
        Self(20)
    }
}

impl Into<usize> for ScientificDigits {
    fn into(self) -> usize {
        self.0 as usize
    }
}

impl TryFrom<Term> for ScientificDigits {
    type Error = TryIntoIntegerError;

    fn try_from(term: Term) -> Result<Self, Self::Error> {
        let scientific_digits_u8: u8 = term.try_into()?;

        if scientific_digits_u8 <= Self::MAX_U8 {
            Ok(Self(scientific_digits_u8))
        } else {
            Err(TryIntoIntegerError::OutOfRange)
        }
    }
}
