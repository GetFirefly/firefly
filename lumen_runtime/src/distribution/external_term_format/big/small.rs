use liblumen_alloc::erts::exception::Exception;
use liblumen_alloc::{Process, Term};

use super::super::u8;

pub fn decode<'a>(process: &Process, bytes: &'a [u8]) -> Result<(Term, &'a [u8]), Exception> {
    let (len_u8, after_len_bytes) = u8::decode(bytes)?;
    let len_usize = len_u8 as usize;

    super::decode(process, after_len_bytes, len_usize)
}
