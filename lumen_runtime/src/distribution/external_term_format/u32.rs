use std::convert::TryInto;
use std::mem;

use liblumen_alloc::erts::exception::Exception;

use crate::distribution::external_term_format::try_split_at;

pub fn decode<'a>(bytes: &'a [u8]) -> Result<(u32, &'a [u8]), Exception> {
    try_split_at(bytes, mem::size_of::<u32>()).map(|(len_bytes, after_len_bytes)| {
        let len_array = len_bytes.try_into().unwrap();
        let len_u32 = u32::from_be_bytes(len_array);

        (len_u32, after_len_bytes)
    })
}
