#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::ptr::NonNull;

use hashbrown::HashMap;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::Term;

#[native_implemented::function(maps:merge/2)]
pub fn result(process: &Process, map1: Term, map2: Term) -> Result<Term, NonNull<ErlangException>> {
    let boxed_map1 = term_try_into_map_or_badmap!(process, map1)?;
    let boxed_map2 = term_try_into_map_or_badmap!(process, map2)?;

    let mut merged: HashMap<Term, Term> =
        HashMap::with_capacity(boxed_map1.len() + boxed_map2.len());

    for (key, value) in boxed_map1.iter() {
        merged.insert(*key, *value);
    }

    for (key, value) in boxed_map2.iter() {
        merged.insert(*key, *value);
    }

    Ok(process.map_from_hash_map(merged))
}
