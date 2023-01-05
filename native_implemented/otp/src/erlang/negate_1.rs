#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::ptr::NonNull;
use anyhow::*;
use num_bigint::BigInt;
use firefly_rt::backtrace::Trace;
use firefly_rt::error::ErlangException;

use firefly_rt::process::Process;
use firefly_rt::term::Term;

/// `-/1` prefix operator.
#[native_implemented::function(erlang:-/1)]
pub fn result(process: &Process, number: Term) -> Result<Term, NonNull<ErlangException>> {
    match number {
        Term::Int(small_integer) => {
            let number_isize: isize = small_integer.into();
            let negated_isize = -number_isize;
            let negated_number: Term = process.integer(negated_isize).unwrap();

            Ok(negated_number)
        }
        Term::BigInt(big_integer) => {
            let big_int: &BigInt = big_integer.as_ref().into();
            let negated_big_int = -big_int;
            let negated_number = process.integer(negated_big_int).unwrap();

            Ok(negated_number)
        }
        Term::Float(float) => {
            let number_f64: f64 = float.into();
            let negated_f64: f64 = -number_f64;
            let negated_number = negated_f64.into();

            Ok(negated_number)
        }
        _ => Err(badarith(
            Trace::capture(),
            Some(anyhow!("number ({}) is neither an integer nor a float", number).into()),
        )
        .into()),
    }
}
