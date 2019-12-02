use std::convert::TryInto;

use num_bigint::BigInt;

use radix_fmt::radix;

use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::otp::erlang::base::Base;

pub fn base_integer_to_string(
    process: &Process,
    base: Term,
    integer: Term,
) -> exception::Result<String> {
    let base: Base = base.try_into().map_err(|_| badarg!(process))?;

    let option_string: Option<String> = match integer.decode()? {
        TypedTerm::SmallInteger(small_integer) => {
            let integer_isize: isize = small_integer.into();

            let (sign, radix) = if integer_isize < 0 {
                let radix = radix(-1 * integer_isize, base.base());

                ("-", radix)
            } else {
                let radix = radix(integer_isize, base.base());
                ("", radix)
            };

            Some(format!("{}{}", sign, radix))
        }
        TypedTerm::BigInteger(big_integer) => {
            let big_int: &BigInt = big_integer.as_ref().into();

            Some(big_int.to_str_radix(base.radix()))
        }
        _ => None,
    };

    match option_string {
        Some(string) => Ok(string),
        None => Err(badarg!(process).into()),
    }
}

pub fn decimal_integer_to_string(process: &Process, integer: Term) -> exception::Result<String> {
    let option_string: Option<String> = match integer.decode()? {
        TypedTerm::SmallInteger(small_integer) => Some(small_integer.to_string()),
        TypedTerm::BigInteger(big_integer) => Some(big_integer.to_string()),
        _ => None,
    };

    match option_string {
        Some(string) => Ok(string),
        None => Err(badarg!(process).into()),
    }
}
