pub mod large;
pub mod small;

use num_bigint::BigInt;

use liblumen_alloc::erts::exception::InternalResult;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::Process;

use super::{sign, try_split_at};

fn decode<'a>(process: &Process, bytes: &'a [u8], len: usize) -> InternalResult<(Term, &'a [u8])> {
    let (sign, after_sign_bytes) = sign::decode(bytes)?;

    try_split_at(after_sign_bytes, len).and_then(|(digits_bytes, after_digits_bytes)| {
        let big_int = BigInt::from_bytes_le(sign, digits_bytes);
        let integer = process.integer(big_int)?;

        Ok((integer, after_digits_bytes))
    })
}
