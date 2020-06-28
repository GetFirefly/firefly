#[cfg(all(not(any(target_arch = "wasm32", feature = "runtime_minimal")), test))]
mod test;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

#[native_implemented::function(tuple_size/1)]
pub fn result(process: &Process, tuple: Term) -> exception::Result<Term> {
    let tuple = term_try_into_tuple!(tuple)?;
    let size = process.integer(tuple.len())?;

    Ok(size)
}
