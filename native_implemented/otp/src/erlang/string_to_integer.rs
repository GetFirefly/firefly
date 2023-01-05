use std::convert::TryInto;

use anyhow::*;
use num_bigint::BigInt;

use firefly_alloc::gc::GcBox;

use firefly_rt::process::Process;
use firefly_rt::term::{OpaqueTerm, Term};

use crate::runtime::base::Base;
use crate::runtime::context;

pub fn base_string_to_integer(
    process: &Process,
    base: Term,
    name: &'static str,
    term: Term,
    string: &str,
) -> Result<OpaqueTerm> {
    let base_base: Base = base.try_into()?;
    let bytes = string.as_bytes();

    match BigInt::parse_bytes(bytes, base_base.radix()) {
        Some(big_int) => {
            let big_integer: BigInteger = big_int.into();
            let gc_box_big_integer = GcBox::new_in(big_integer, process)?;
            Ok(gc_box_big_integer.into())
        }
        None => Err(anyhow!("{} is not in base ({})", context::string(name, term), base).into()),
    }
}

pub fn decimal_string_to_integer(
    process: &Process,
    name: &'static str,
    term: Term,
    string: &str,
) -> InternalResult<Term> {
    match BigInt::parse_bytes(string.as_bytes(), 10) {
        Some(big_int) => Ok(process.integer(big_int).unwrap()),
        None => Err(anyhow!("{} is not base 10", context::string(name, term)).into()),
    }
}
