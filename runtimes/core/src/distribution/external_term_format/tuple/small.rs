use liblumen_alloc::erts::exception::InternalResult;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::erts::Process;

use super::super::u8;

pub fn decode<'a>(
    process: &Process,
    safe: bool,
    bytes: &'a [u8],
) -> InternalResult<(Term, &'a [u8])> {
    let (len_u8, after_len_bytes) = u8::decode(bytes)?;

    super::decode(process, safe, after_len_bytes, len_u8 as usize)
}
