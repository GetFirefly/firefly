// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;

use anyhow::*;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use native_implemented_function::native_implemented_function;

#[native_implemented_function(float/1)]
pub fn result(process: &Process, number: Term) -> exception::Result<Term> {
    if number.is_boxed_float() {
        Ok(number)
    } else {
        let f: f64 = number
            .try_into()
            .with_context(|| term_is_not_number!(number))?;

        process.float(f).map_err(From::from)
    }
}
