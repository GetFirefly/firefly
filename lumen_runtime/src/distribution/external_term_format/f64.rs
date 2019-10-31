use std::convert::TryInto;
use std::mem;

use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception::Exception;

pub fn decode(bytes: &[u8]) -> Result<(f64, &[u8]), Exception> {
    const F64_BYTE_LEN: usize = mem::size_of::<f64>();

    if F64_BYTE_LEN <= bytes.len() {
        let (f64_bytes, after_f64_bytes) = bytes.split_at(F64_BYTE_LEN);
        let f64_array = f64_bytes.try_into().unwrap();
        let f = f64::from_be_bytes(f64_array);

        Ok((f, after_f64_bytes))
    } else {
        Err(badarg!().into())
    }
}
