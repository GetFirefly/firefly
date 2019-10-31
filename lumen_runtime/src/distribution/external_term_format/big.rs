pub mod large;
pub mod small;

use num_bigint::BigInt;

use liblumen_alloc::erts::exception::Exception;
use liblumen_alloc::{badarg, Process, Term};

use super::sign;

fn decode<'a>(
    process: &Process,
    bytes: &'a [u8],
    len: usize,
) -> Result<(Term, &'a [u8]), Exception> {
    let (sign, after_sign_bytes) = sign::decode(bytes)?;

    if len <= after_sign_bytes.len() {
        let (digits_bytes, after_digits_bytes) = after_sign_bytes.split_at(len);
        let big_int = BigInt::from_bytes_le(sign, digits_bytes);
        let integer = process.integer(big_int)?;

        Ok((integer, after_digits_bytes))
    } else {
        Err(badarg!().into())
    }
}
