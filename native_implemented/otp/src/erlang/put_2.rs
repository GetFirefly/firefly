#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Term;

#[native_implemented::function(erlang:put/2)]
pub fn result(process: &Process, key: Term, value: Term) -> exception::Result<Term> {
    process.put(key, value).map_err(|alloc| alloc.into())
}
