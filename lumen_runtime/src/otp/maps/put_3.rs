// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;

use anyhow::*;

use liblumen_alloc::erts::exception::{self, *};
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use lumen_runtime_macros::native_implemented_function;

#[native_implemented_function(put/3)]
pub fn native(process: &Process, key: Term, value: Term, map: Term) -> exception::Result<Term> {
    let boxed_map: Boxed<Map> = map
        .try_into()
        .with_context(|| format!("map ({}) is not a map", map))
        .map_err(|source| badmap(process, map, source.into()))?;

    match boxed_map.put(key, value) {
        Some(hash_map) => Ok(process.map_from_hash_map(hash_map)?),
        None => Ok(map),
    }
}
