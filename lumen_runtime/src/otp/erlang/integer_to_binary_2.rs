// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;

use num_bigint::BigInt;

use radix_fmt::radix;

use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::{Term, TypedTerm};

use lumen_runtime_macros::native_implemented_function;

use crate::otp::erlang::base::Base;

#[native_implemented_function(integer_to_binary/2)]
pub fn native(process: &Process, integer: Term, base: Term) -> exception::Result {
    let base: Base = base.try_into()?;

    let option_string: Option<String> = match integer.to_typed_term().unwrap() {
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
        TypedTerm::Boxed(boxed) => match boxed.to_typed_term().unwrap() {
            TypedTerm::BigInteger(big_integer) => {
                let big_int: &BigInt = big_integer.as_ref().into();

                Some(big_int.to_str_radix(base.radix()))
            }
            _ => None,
        },
        _ => None,
    };

    match option_string {
        Some(string) => process
            .binary_from_str(&string)
            .map_err(|alloc| alloc.into()),
        None => Err(badarg!().into()),
    }
}
