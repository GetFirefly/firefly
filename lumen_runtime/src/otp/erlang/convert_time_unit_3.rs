// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;

use num_bigint::BigInt;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Term;(

use lumen_runtime_macros::native_implemented_function;

use crate::time;

#[native_implemented_function(convert_time_unit/3)]
pub fn native(process: &Process, time: Term, from_unit: Term, to_unit: Term) -> exception::Result {
    let time_big_int: BigInt = time.try_into()?;
    let from_unit_unit: time::Unit = from_unit.try_into()?;
    let to_unit_unit: time::Unit = to_unit.try_into()?;
    let converted_big_int = time::convert(time_big_int, from_unit_unit, to_unit_unit);
    let converted_term = process.integer(converted_big_int)?;

    Ok(converted_term)
}
