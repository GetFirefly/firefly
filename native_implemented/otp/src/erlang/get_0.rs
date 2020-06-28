#[cfg(all(not(any(target_arch = "wasm32", feature = "runtime_minimal")), test))]
mod test;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

#[native_implemented::function(get/0)]
pub fn result(process: &Process) -> exception::Result<Term> {
    process.get_entries().map_err(|alloc| alloc.into())
}
