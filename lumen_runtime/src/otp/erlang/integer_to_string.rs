use std::convert::TryInto;

use anyhow::*;
use num_bigint::BigInt;
use radix_fmt::radix;

use liblumen_alloc::erts::exception::InternalResult;
use liblumen_alloc::erts::term::prelude::*;

use crate::otp::erlang::base::Base;

pub fn base_integer_to_string(base: Term, integer: Term) -> InternalResult<String> {
    let base: Base = base.try_into()?;

    match integer.decode()? {
        TypedTerm::SmallInteger(small_integer) => {
            let integer_isize: isize = small_integer.into();

            let (sign, radix) = if integer_isize < 0 {
                let radix = radix(-1 * integer_isize, base.base());

                ("-", radix)
            } else {
                let radix = radix(integer_isize, base.base());
                ("", radix)
            };

            Ok(format!("{}{}", sign, radix))
        }
        TypedTerm::BigInteger(big_integer) => {
            let big_int: &BigInt = big_integer.as_ref().into();

            Ok(big_int.to_str_radix(base.radix()))
        }
        _ => Err(TypeError)
            .context(format!("integer ({}) is not an integer", integer))
            .map_err(From::from),
    }
}

pub fn decimal_integer_to_string(integer: Term) -> InternalResult<String> {
    match integer.decode()? {
        TypedTerm::SmallInteger(small_integer) => Ok(small_integer.to_string()),
        TypedTerm::BigInteger(big_integer) => Ok(big_integer.to_string()),
        _ => Err(TypeError)
            .context(format!("integer ({}) is not an integer", integer))
            .map_err(From::from),
    }
}
