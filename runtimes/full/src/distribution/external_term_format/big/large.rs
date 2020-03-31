use liblumen_alloc::erts::exception::InternalResult;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::erts::Process;

use super::super::u32;

pub fn decode<'a>(process: &Process, bytes: &'a [u8]) -> InternalResult<(Term, &'a [u8])> {
    let (len_u32, after_len_bytes) = u32::decode(bytes)?;
    let len_usize = len_u32 as usize;

    super::decode(process, after_len_bytes, len_usize)
}
