#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

#[native_implemented::function(erlang:get/0)]
pub fn result(process: &Process) -> exception::Result<Term> {
    process.get_entries().map_err(|alloc| alloc.into())
}
