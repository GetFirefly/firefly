// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;

use liblumen_alloc::badarith;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Term;

use lumen_runtime_macros::native_implemented_function;

/// `//2` infix operator.  Unlike `+/2`, `-/2` and `*/2` always promotes to `float` returns the
/// `float`.
#[native_implemented_function(/ /2)]
pub fn native(process: &Process, dividend: Term, divisor: Term) -> exception::Result<Term> {
    let dividend_f64: f64 = dividend.try_into().map_err(|_| badarith!())?;
    let divisor_f64: f64 = divisor.try_into().map_err(|_| badarith!())?;

    if divisor_f64 == 0.0 {
        Err(badarith!().into())
    } else {
        let quotient_f64 = dividend_f64 / divisor_f64;
        let quotient_term = process.float(quotient_f64)?;

        Ok(quotient_term)
    }
}
