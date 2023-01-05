use std::ptr::NonNull;

use anyhow::*;
use num_bigint::BigInt;
use firefly_rt::backtrace::Trace;

use firefly_rt::process::Process;
use firefly_rt::error::ErlangException;
use firefly_rt::term::Term;

/// `bnot/1` prefix operator.
#[native_implemented::function(erlang:bnot/1)]
pub fn result(process: &Process, integer: Term) -> Result<Term, NonNull<ErlangException>> {
    match integer {
        Term::Int(small_integer) => {
            let integer_isize: isize = small_integer.into();
            let output = !integer_isize;
            let output_term = process.integer(output).unwrap();

            Ok(output_term)
        }
        Term::BigInt(big_integer) => {
            let big_int: &BigInt = big_integer.as_ref().into();
            let output_big_int = !big_int;
            let output_term = process.integer(output_big_int).unwrap();

            Ok(output_term)
        }
        _ => Err(badarith(
            Trace::capture(),
            Some(anyhow!("integer ({}) is not an integer", integer).into()),
        )
        .into()),
    }
}
