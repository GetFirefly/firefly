// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;

use anyhow::*;
use hashbrown::HashMap;

use liblumen_alloc::erts::exception::{self, *};
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use lumen_runtime_macros::native_implemented_function;

#[native_implemented_function(merge/2)]
pub fn native(process: &Process, map1: Term, map2: Term) -> exception::Result<Term> {
    let boxed_map1: Boxed<Map> = map1
        .try_into()
        .with_context(|| format!("map1 ({}) is not a map", map1))
        .map_err(|source| badmap(process, map1, source.into()))?;
    let boxed_map2: Boxed<Map> = map2
        .try_into()
        .with_context(|| format!("map2 ({}) is not a map", map2))
        .map_err(|source| badmap(process, map2, source.into()))?;

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
