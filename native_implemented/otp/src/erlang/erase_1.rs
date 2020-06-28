#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Term;

#[native_implemented::function(erase/1)]
pub fn result(process: &Process, key: Term) -> Term {
    process.erase_value_from_key(key)
}
