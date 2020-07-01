use std::convert::TryInto;
use std::mem;

use liblumen_alloc::erts::exception::InternalResult;

use crate::distribution::external_term_format::try_split_at;

pub fn decode<'a>(bytes: &'a [u8]) -> InternalResult<(u16, &'a [u8])> {
    try_split_at(bytes, mem::size_of::<u16>()).map(|(len_bytes, after_len_bytes)| {
        let len_array = len_bytes.try_into().unwrap();
        let len_u16 = u16::from_be_bytes(len_array);

        (len_u16, after_len_bytes)
    })
}
