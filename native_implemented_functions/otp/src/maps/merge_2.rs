#[cfg(all(not(any(target_arch = "wasm32", feature = "runtime_minimal")), test))]
mod test;

use hashbrown::HashMap;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

#[native_implemented::function(merge/2)]
pub fn result(process: &Process, map1: Term, map2: Term) -> exception::Result<Term> {
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

    process.map_from_hash_map(merged).map_err(From::from)
}
