#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Term;

#[native_implemented::function(erlang:get_keys/1)]
pub fn result(process: &Process, value: Term) -> exception::Result<Term> {
    process
        .get_keys_from_value(value)
        .map_err(|alloc| alloc.into())
}
