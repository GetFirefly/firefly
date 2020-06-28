#[cfg(all(not(any(target_arch = "wasm32", feature = "runtime_minimal")), test))]
mod test;

use liblumen_alloc::atom;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

#[native_implemented::function(take/2)]
pub fn result(process: &Process, key: Term, map: Term) -> exception::Result<Term> {
    let boxed_map = term_try_into_map_or_badmap!(process, map)?;

    let result = match boxed_map.take(key) {
        Some((value, hash_map)) => {
            let map = process.map_from_hash_map(hash_map)?;
            process.tuple_from_slice(&[value, map])?
        }
        None => atom!("error"),
    };

    Ok(result)
}
