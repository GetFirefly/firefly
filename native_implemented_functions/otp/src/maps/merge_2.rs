// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use hashbrown::HashMap;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use native_implemented_function::native_implemented_function;

#[native_implemented_function(merge/2)]
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
