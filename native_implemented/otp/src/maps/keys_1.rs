#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

#[native_implemented::function(maps:keys/1)]
pub fn result(process: &Process, map: Term) -> exception::Result<Term> {
    let boxed_map = term_try_into_map_or_badmap!(process, map)?;
    let keys = boxed_map.keys();
    let list = process.list_from_slice(&keys)?;

    Ok(list)
}
