#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Term;

#[native_implemented::function(erlang:put/2)]
pub fn result(process: &Process, key: Term, value: Term) -> Term {
    process.put(key, value)
}
