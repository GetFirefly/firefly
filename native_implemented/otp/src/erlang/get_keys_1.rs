#[cfg(all(not(any(target_arch = "wasm32", feature = "runtime_minimal")), test))]
mod test;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Term;

#[native_implemented::function(get_keys/1)]
pub fn result(process: &Process, value: Term) -> exception::Result<Term> {
    process
        .get_keys_from_value(value)
        .map_err(|alloc| alloc.into())
}
