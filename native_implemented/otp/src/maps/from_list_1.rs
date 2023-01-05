use std::ptr::NonNull;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::{Map, Term};

#[native_implemented::function(maps:from_list/1)]
pub fn result(process: &Process, list: Term) -> Result<Term, NonNull<ErlangException>> {
    let hash_map = Map::from_list(list)?;
    let map = process.map_from_hash_map(hash_map);

    Ok(map)
}
