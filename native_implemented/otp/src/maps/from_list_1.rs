#[cfg(all(not(any(target_arch = "wasm32", feature = "runtime_minimal")), test))]
mod test;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

#[native_implemented::function(from_list/1)]
pub fn result(process: &Process, list: Term) -> exception::Result<Term> {
    let hash_map = Map::from_list(list)?;
    let map = process.map_from_hash_map(hash_map)?;

    Ok(map)
}
