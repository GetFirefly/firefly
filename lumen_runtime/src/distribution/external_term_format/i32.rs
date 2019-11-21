use std::convert::TryInto;
use std::mem;

use liblumen_alloc::erts::exception::Exception;

use super::try_split_at;

pub fn decode(bytes: &[u8]) -> Result<(i32, &[u8]), Exception> {
    try_split_at(bytes, mem::size_of::<i32>()).map(|(len_bytes, after_len_bytes)| {
        let len_array = len_bytes.try_into().unwrap();
        let len_i32 = i32::from_be_bytes(len_array);

        (len_i32, after_len_bytes)
    })
}
