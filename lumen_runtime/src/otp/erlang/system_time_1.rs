// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;

use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Term;

use lumen_runtime_macros::native_implemented_function;

use crate::time::{system, Unit};

#[native_implemented_function(system_time/1)]
pub fn native(process: &Process, unit: Term) -> exception::Result<Term> {
    let unit_unit: Unit = unit.try_into().map_err(|_| badarg!())?;
    let big_int = system::time(unit_unit);
    let term = process.integer(big_int)?;

    Ok(term)
}
