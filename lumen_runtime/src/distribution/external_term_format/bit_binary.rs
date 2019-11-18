use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception::Exception;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::erts::Process;

use super::{u32, u8};

pub fn decode<'a>(process: &Process, bytes: &'a [u8]) -> Result<(Term, &'a [u8]), Exception> {
    let (len_u32, after_len_bytes) = u32::decode(bytes)?;
    let len_usize = len_u32 as usize;

    let (partial_byte_bit_len, after_partial_byte_bit_len_bytes) = u8::decode(after_len_bytes)?;
    assert!(0 < partial_byte_bit_len);

    if len_usize <= after_partial_byte_bit_len_bytes.len() {
        let (data_bytes, after_data_bytes) = after_partial_byte_bit_len_bytes.split_at(len_usize);
        let original = process.binary_from_bytes(data_bytes)?;
        let subbinary =
            process.subbinary_from_original(original, 0, 0, len_usize - 1, partial_byte_bit_len)?;

        Ok((subbinary, after_data_bytes))
    } else {
        Err(badarg!().into())
    }
}
