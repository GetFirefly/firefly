#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use liblumen_alloc::atom;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

#[native_implemented::function(maps:take/2)]
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
