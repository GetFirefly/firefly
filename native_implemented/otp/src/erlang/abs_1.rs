use std::cmp::Ordering;
use std::ptr::NonNull;

use anyhow::*;
use num_bigint::BigInt;
use num_traits::Zero;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::{Term, TypeError};

#[native_implemented::function(erlang:abs/1)]
fn result(process: &Process, number: Term) -> Result<Term, NonNull<ErlangException>> {
    match number {
        Term::Int(small_integer) => {
            let i: isize = small_integer.into();

            let abs_number = if i < 0 {
                let positive = -i;
                process.integer(positive).unwrap()
            } else {
                number
            };

            Ok(abs_number)
        }
        Term::BigInt(big_integer) => {
            let big_int: &BigInt = big_integer.as_ref().into();
            let zero_big_int: &BigInt = &Zero::zero();

            let abs_number: Term = if big_int < zero_big_int {
                let positive_big_int: BigInt = -1 * big_int;

                process.integer(positive_big_int).unwrap()
            } else {
                number
            };

            Ok(abs_number)
        }
        Term::Float(float) => {
            let f: f64 = float.into();

            let abs_number = match f.partial_cmp(&0.0).unwrap() {
                Ordering::Less => f.abs().into(),
                _ => number,
            };

            Ok(abs_number)
        }
        _ => Err(TypeError)
            .context(term_is_not_number!(number))
            .map_err(From::from),
    }
}
