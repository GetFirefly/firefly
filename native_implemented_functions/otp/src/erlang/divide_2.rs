// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;

use anyhow::*;

use liblumen_alloc::erts::exception::{self, *};
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Term;

use native_implemented_function::native_implemented_function;

/// `//2` infix operator.  Unlike `+/2`, `-/2` and `*/2` always promotes to `float` returns the
/// `float`.
#[native_implemented_function(/ /2)]
pub fn result(process: &Process, dividend: Term, divisor: Term) -> exception::Result<Term> {
    let dividend_f64: f64 = dividend.try_into().map_err(|_| {
        badarith(anyhow!("dividend ({}) cannot be promoted to a float", dividend).into())
    })?;
    let divisor_f64: f64 = divisor.try_into().map_err(|_| {
        badarith(anyhow!("divisor ({}) cannot be promoted to a float", divisor).into())
    })?;

    if divisor_f64 == 0.0 {
        Err(badarith(anyhow!("divisor ({}) cannot be zero", divisor).into()).into())
    } else {
        let quotient_f64 = dividend_f64 / divisor_f64;
        let quotient_term = process.float(quotient_f64)?;

        Ok(quotient_term)
    }
}
