#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Term;

#[native_implemented::function(get/1)]
pub fn result(process: &Process, key: Term) -> Term {
    process.get_value_from_key(key)
}
