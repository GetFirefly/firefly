use liblumen_alloc::erts::exception::Exception;
use liblumen_alloc::{Process, Term};

use super::i32;

pub fn decode<'a>(process: &Process, bytes: &'a [u8]) -> Result<(Term, &'a [u8]), Exception> {
    let (integer_i32, after_integer_bytes) = i32::decode(bytes)?;
    let integer = process.integer(integer_i32)?;

    Ok((integer, after_integer_bytes))
}
