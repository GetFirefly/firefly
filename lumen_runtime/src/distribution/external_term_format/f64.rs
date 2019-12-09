use std::convert::TryInto;
use std::mem;

use liblumen_alloc::erts::exception::InternalResult;

use super::try_split_at;

pub fn decode(bytes: &[u8]) -> InternalResult<(f64, &[u8])> {
    try_split_at(bytes, mem::size_of::<f64>()).map(|(f64_bytes, after_f64_bytes)| {
        let f64_array = f64_bytes.try_into().unwrap();
        let f = f64::from_be_bytes(f64_array);

        (f, after_f64_bytes)
    })
}
