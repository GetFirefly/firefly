// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::cmp::Ordering;

use anyhow::*;
use num_bigint::BigInt;
use num_traits::Zero;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

#[native_implemented::function(abs/1)]
fn result(process: &Process, number: Term) -> exception::Result<Term> {
    match number.decode()? {
        TypedTerm::SmallInteger(small_integer) => {
            let i: isize = small_integer.into();

            let abs_number = if i < 0 {
                let positive = -i;
                process.integer(positive)?
            } else {
                number
            };

            Ok(abs_number)
        }
        TypedTerm::BigInteger(big_integer) => {
            let big_int: &BigInt = big_integer.as_ref().into();
            let zero_big_int: &BigInt = &Zero::zero();

            let abs_number: Term = if big_int < zero_big_int {
                let positive_big_int: BigInt = -1 * big_int;

                process.integer(positive_big_int)?
            } else {
                number
            };

            Ok(abs_number)
        }
        TypedTerm::Float(float) => {
            let f: f64 = float.into();

            let abs_number = match f.partial_cmp(&0.0).unwrap() {
                Ordering::Less => {
                    let positive_f = f.abs();

                    process.float(positive_f).unwrap()
                }
                _ => number,
            };

            Ok(abs_number)
        }
        _ => Err(TypeError)
            .context(term_is_not_number!(number))
            .map_err(From::from),
    }
}
