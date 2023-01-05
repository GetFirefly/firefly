use std::convert::{TryFrom, TryInto};

use firefly_number::TryIntoIntegerError;
use firefly_rt::term::Term;

pub struct Compression(pub u8);

impl Compression {
    const MIN_U8: u8 = 0;
    const MAX_U8: u8 = 9;
}

impl Default for Compression {
    fn default() -> Self {
        // Default level when option compressed is provided.
        Self(6)
    }
}

impl TryFrom<Term> for Compression {
    type Error = TryFromTermError;

    fn try_from(term: Term) -> Result<Self, Self::Error> {
        let term_u8: u8 = term.try_into()?;

        if Self::MIN_U8 <= term_u8 && term_u8 <= Self::MAX_U8 {
            Ok(Self(term_u8))
        } else {
            Err(TryFromTermError::OutOfRange)
        }
    }
}

pub enum TryFromTermError {
    OutOfRange,
    Type,
}

impl From<TryIntoIntegerError> for TryFromTermError {
    fn from(error: TryIntoIntegerError) -> Self {
        match error {
            TryIntoIntegerError::Type => TryFromTermError::Type,
            TryIntoIntegerError::OutOfRange => TryFromTermError::OutOfRange,
        }
    }
}
