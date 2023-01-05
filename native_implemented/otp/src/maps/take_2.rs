#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::ptr::NonNull;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::{atoms, Term};

#[native_implemented::function(maps:take/2)]
pub fn result(process: &Process, key: Term, map: Term) -> Result<Term, NonNull<ErlangException>> {
    let boxed_map = term_try_into_map_or_badmap!(process, map)?;

    let result = match boxed_map.take(key) {
        Some((value, hash_map)) => {
            let map = process.map_from_hash_map(hash_map);
            process.tuple_term_from_term_slice(&[value, map])
        }
        None => atoms::Error.into(),
    };

    Ok(result)
}
