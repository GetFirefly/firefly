#[cfg(all(not(any(target_arch = "wasm32", feature = "runtime_minimal")), test))]
mod test;

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Term;

#[native_implemented::function(erase/1)]
pub fn result(process: &Process, key: Term) -> Term {
    process.erase_value_from_key(key)
}
