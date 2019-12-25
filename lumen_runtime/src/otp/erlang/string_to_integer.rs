use std::convert::TryInto;

use anyhow::*;
use num_bigint::BigInt;

use liblumen_alloc::erts::exception::InternalResult;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Term;

use crate::otp::erlang::base::Base;

pub fn base_string_to_integer(process: &Process, base: Term, string: &str) -> InternalResult<Term> {
    let base_base: Base = base.try_into()?;
    let bytes = string.as_bytes();

    match BigInt::parse_bytes(bytes, base_base.radix()) {
        Some(big_int) => process.integer(big_int).map_err(|error| error.into()),
        None => Err(anyhow!("string ({}) is not in base ({})", string, base).into()),
    }
}

pub fn decimal_string_to_integer(process: &Process, string: &str) -> InternalResult<Term> {
    match BigInt::parse_bytes(string.as_bytes(), 10) {
        Some(big_int) => process.integer(big_int).map_err(|error| error.into()),
        None => Err(anyhow!("string ({}) is not base 10", string).into()),
    }
}
