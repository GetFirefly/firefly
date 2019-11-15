use liblumen_alloc::erts::exception::Exception;
use liblumen_alloc::erts::Process;
use liblumen_alloc::erts::term::prelude::*;

use super::super::u32;

pub fn decode<'a>(process: &Process, bytes: &'a [u8]) -> Result<(Term, &'a [u8]), Exception> {
    let (len_u32, after_len_bytes) = u32::decode(bytes)?;
    let len_usize = len_u32 as usize;

    super::decode(process, after_len_bytes, len_usize)
}
