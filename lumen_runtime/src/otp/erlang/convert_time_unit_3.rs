// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;

use anyhow::*;
use num_bigint::BigInt;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Term;

use lumen_runtime_macros::native_implemented_function;

use crate::time;

#[native_implemented_function(convert_time_unit/3)]
pub fn native(
    process: &Process,
    time: Term,
    from_unit: Term,
    to_unit: Term,
) -> exception::Result<Term> {
    let time_big_int: BigInt = time.try_into().context("time must be an integer")?;
    let from_unit_unit: time::Unit = from_unit
        .try_into()
        .context("from_must must be a time unit")?;
    let to_unit_unit: time::Unit = to_unit.try_into().context("to_unit must be a time unit")?;
    let converted_big_int = time::convert(time_big_int, from_unit_unit, to_unit_unit);
    let converted_term = process
        .integer(converted_big_int)
        .map_err(|_| badarg!(process))?;

    Ok(converted_term)
}
