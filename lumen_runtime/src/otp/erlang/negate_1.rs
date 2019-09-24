// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use num_bigint::BigInt;

use liblumen_alloc::badarith;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::{Term, TypedTerm};

use lumen_runtime_macros::native_implemented_function;

/// `-/1` prefix operator.
#[native_implemented_function(-/1)]
pub fn native(process: &Process, number: Term) -> exception::Result {
    let option_negated = match number.to_typed_term().unwrap() {
        TypedTerm::SmallInteger(small_integer) => {
            let number_isize: isize = small_integer.into();
            let negated_isize = -number_isize;
            let negated_number: Term = process.integer(negated_isize)?;

            Some(negated_number)
        }
        TypedTerm::Boxed(boxed) => match boxed.to_typed_term().unwrap() {
            TypedTerm::BigInteger(big_integer) => {
                let big_int: &BigInt = big_integer.as_ref().into();
                let negated_big_int = -big_int;
                let negated_number = process.integer(negated_big_int)?;

                Some(negated_number)
            }
            TypedTerm::Float(float) => {
                let number_f64: f64 = float.into();
                let negated_f64: f64 = -number_f64;
                let negated_number = process.float(negated_f64)?;

                Some(negated_number)
            }
            _ => None,
        },
        _ => None,
    };

    match option_negated {
        Some(negated) => Ok(negated),
        None => Err(badarith!().into()),
    }
}
