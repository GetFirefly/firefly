// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use num_bigint::BigInt;

use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use lumen_runtime_macros::native_implemented_function;

#[native_implemented_function(ceil/1)]
pub fn native(process: &Process, number: Term) -> exception::Result {
    let option_ceil = match number.to_typed_term().unwrap() {
        TypedTerm::SmallInteger(_) => Some(number),
        TypedTerm::Boxed(boxed) => {
            match boxed.to_typed_term().unwrap() {
                TypedTerm::BigInteger(_) => Some(number),
                TypedTerm::Float(float) => {
                    let inner: f64 = float.into();
                    let ceil_inner = inner.ceil();

                    // skip creating a BigInt if float can fit in small integer.
                    let ceil_term = if (SmallInteger::MIN_VALUE as f64).max(Float::INTEGRAL_MIN)
                        <= ceil_inner
                        && ceil_inner <= (SmallInteger::MAX_VALUE as f64).min(Float::INTEGRAL_MAX)
                    {
                        process.integer(ceil_inner as isize)?
                    } else {
                        let ceil_string = ceil_inner.to_string();
                        let ceil_bytes = ceil_string.as_bytes();
                        let big_int = BigInt::parse_bytes(ceil_bytes, 10).unwrap();

                        process.integer(big_int)?
                    };

                    Some(ceil_term)
                }
                _ => None,
            }
        }
        _ => None,
    };

    match option_ceil {
        Some(ceil) => Ok(ceil),
        None => Err(badarg!().into()),
    }
}
