use std::convert::TryInto;

use anyhow::*;
use num_bigint::BigInt;

use liblumen_alloc::erts::exception::InternalResult;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Term;

use crate::erlang::base::Base;
use crate::runtime::context;

pub fn base_string_to_integer(
    process: &Process,
    base: Term,
    name: &'static str,
    quote: char,
    string: &str,
) -> InternalResult<Term> {
    let base_base: Base = base.try_into()?;
    let bytes = string.as_bytes();

    match BigInt::parse_bytes(bytes, base_base.radix()) {
        Some(big_int) => process.integer(big_int).map_err(|error| error.into()),
        None => Err(anyhow!(
            "{} is not in base ({})",
            context::string(name, quote, string),
            base
        )
        .into()),
    }
}

pub fn decimal_string_to_integer(
    process: &Process,
    name: &'static str,
    quote: char,
    value: &str,
) -> InternalResult<Term> {
    match BigInt::parse_bytes(value.as_bytes(), 10) {
        Some(big_int) => process.integer(big_int).map_err(|error| error.into()),
        None => Err(anyhow!("{} is not base 10", context::string(name, quote, value)).into()),
    }
}
