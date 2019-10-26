use std::convert::TryInto;

use num_bigint::BigInt;

use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Term;

use crate::otp::erlang::base::Base;

pub fn base_string_to_integer(process: &Process, base: Term, string: &str) -> exception::Result {
    let base: Base = base.try_into()?;
    let bytes = string.as_bytes();

    match BigInt::parse_bytes(bytes, base.radix()) {
        Some(big_int) => process.integer(big_int).map_err(|error| error.into()),
        None => Err(badarg!().into()),
    }
}

pub fn decimal_string_to_integer(process: &Process, string: &str) -> exception::Result {
    match BigInt::parse_bytes(string.as_bytes(), 10) {
        Some(big_int) => process.integer(big_int).map_err(|error| error.into()),
        None => Err(badarg!().into()),
    }
}
