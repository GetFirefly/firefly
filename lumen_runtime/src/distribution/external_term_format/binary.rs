use liblumen_alloc::erts::exception::InternalResult;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::erts::Process;

use super::u32;
use crate::distribution::external_term_format::try_split_at;

pub fn decode<'a>(process: &Process, bytes: &'a [u8]) -> InternalResult<(Term, &'a [u8])> {
    let (len_u32, after_len_bytes) = u32::decode(bytes)?;
    let len_usize = len_u32 as usize;

    try_split_at(after_len_bytes, len_usize).and_then(|(data_bytes, after_data_bytes)| {
        let binary_term = process.binary_from_bytes(data_bytes)?;

        Ok((binary_term, after_data_bytes))
    })
}
