use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception::Exception;
use liblumen_alloc::erts::term::Term;
use liblumen_alloc::erts::Process;

use super::u32;

pub fn decode<'a>(process: &Process, bytes: &'a [u8]) -> Result<(Term, &'a [u8]), Exception> {
    let (len_u32, after_len_bytes) = u32::decode(bytes)?;
    let len_usize = len_u32 as usize;

    if len_usize <= after_len_bytes.len() {
        let (data_bytes, after_data_bytes) = after_len_bytes.split_at(len_usize);
        let binary_term = process.binary_from_bytes(data_bytes)?;

        Ok((binary_term, after_data_bytes))
    } else {
        Err(badarg!().into())
    }
}
