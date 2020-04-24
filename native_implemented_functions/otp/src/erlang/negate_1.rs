// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use anyhow::*;
use num_bigint::BigInt;

use liblumen_alloc::erts::exception::{self, *};
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use native_implemented_function::native_implemented_function;

/// `-/1` prefix operator.
#[native_implemented_function(-/1)]
pub fn result(process: &Process, number: Term) -> exception::Result<Term> {
    match number.decode().unwrap() {
        TypedTerm::SmallInteger(small_integer) => {
            let number_isize: isize = small_integer.into();
            let negated_isize = -number_isize;
            let negated_number: Term = process.integer(negated_isize)?;

            Ok(negated_number)
        }
        TypedTerm::BigInteger(big_integer) => {
            let big_int: &BigInt = big_integer.as_ref().into();
            let negated_big_int = -big_int;
            let negated_number = process.integer(negated_big_int)?;

            Ok(negated_number)
        }
        TypedTerm::Float(float) => {
            let number_f64: f64 = float.into();
            let negated_f64: f64 = -number_f64;
            let negated_number = process.float(negated_f64)?;

            Ok(negated_number)
        }
        _ => Err(
            badarith(anyhow!("number ({}) is neither an integer nor a float", number).into())
                .into(),
        ),
    }
}
