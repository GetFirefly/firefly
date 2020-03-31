use anyhow::*;
use num_bigint::Sign;

use liblumen_alloc::erts::exception::InternalResult;
use liblumen_alloc::erts::term::prelude::*;

use super::u8;

pub fn decode(bytes: &[u8]) -> InternalResult<(Sign, &[u8])> {
    let (sign_u8, after_sign_bytes) = u8::decode(bytes)?;
    let sign = byte_try_into_sign(sign_u8)?;

    Ok((sign, after_sign_bytes))
}

fn byte_try_into_sign(byte: u8) -> InternalResult<Sign> {
    match byte {
        0 => Ok(Sign::Plus),
        1 => Ok(Sign::Minus),
        _ => Err(TryIntoIntegerError::OutOfRange)
            .context("sign byte can only be 0 for positive and 1 for minus")
            .map_err(|error| error.into()),
    }
}
