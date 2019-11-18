// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::cmp::Ordering;

use num_bigint::BigInt;
use num_traits::Zero;

use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use lumen_runtime_macros::native_implemented_function;

#[native_implemented_function(abs/1)]
pub fn native(process: &Process, number: Term) -> exception::Result<Term> {
    let option_abs = match number.decode().unwrap() {
        TypedTerm::SmallInteger(small_integer) => {
            let i: isize = small_integer.into();

            if i < 0 {
                let positive = -i;
                let abs_number = process.integer(positive)?;

                Some(abs_number)
            } else {
                Some(number)
            }
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

            Some(abs_number)
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

            Some(abs_number)
        }
        _ => None,
    };

    match option_abs {
        Some(abs) => Ok(abs),
        None => Err(badarg!().into()),
    }
}
