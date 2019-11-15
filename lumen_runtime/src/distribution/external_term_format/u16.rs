use std::convert::TryInto;
use std::mem;

use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception::Exception;

pub fn decode(bytes: &[u8]) -> Result<(u16, &[u8]), Exception> {
    const U16_BYTE_LEN: usize = mem::size_of::<u16>();

    if U16_BYTE_LEN <= bytes.len() {
        let (len_bytes, after_len_bytes) = bytes.split_at(U16_BYTE_LEN);
        let len_array = len_bytes.try_into().unwrap();
        let len_u16 = u16::from_be_bytes(len_array);

        Ok((len_u16, after_len_bytes))
    } else {
        Err(badarg!().into())
    }
}
