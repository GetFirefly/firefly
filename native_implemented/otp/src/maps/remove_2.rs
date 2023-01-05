#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::ptr::NonNull;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::Term;

#[native_implemented::function(maps:remove/2)]
pub fn result(process: &Process, key: Term, map: Term) -> Result<Term, NonNull<ErlangException>> {
    let boxed_map = term_try_into_map_or_badmap!(process, map)?;

    match boxed_map.remove(key) {
        Some(hash_map) => Ok(process.map_from_hash_map(hash_map)),
        None => Ok(map),
    }
}
