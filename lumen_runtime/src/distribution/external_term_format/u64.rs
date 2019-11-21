use std::convert::TryInto;
use std::mem;

use liblumen_alloc::erts::exception::Exception;

use super::try_split_at;

pub fn decode(bytes: &[u8]) -> Result<(u64, &[u8]), Exception> {
    try_split_at(bytes, mem::size_of::<u64>()).map(|(len_bytes, after_len_bytes)| {
        let len_array = len_bytes.try_into().unwrap();
        let len_u64 = u64::from_be_bytes(len_array);

        (len_u64, after_len_bytes)
    })
}
