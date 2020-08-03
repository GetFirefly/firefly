use liblumen_alloc::erts::exception::InternalResult;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::erts::Process;

use super::i32;

pub fn decode<'a>(process: &Process, bytes: &'a [u8]) -> InternalResult<(Term, &'a [u8])> {
    let (integer_i32, after_integer_bytes) = i32::decode(bytes)?;
    let integer = process.integer(integer_i32)?;

    Ok((integer, after_integer_bytes))
}
