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

#[native_implemented_function(keys/1)]
pub fn native(process: &Process, map: Term) -> exception::Result<Term> {
    let boxed_map: Boxed<Map> = map
        .try_into()
        .with_context(|| format!("map ({}) is not a map", map))
        .map_err(|source| badmap(process, map, source.into()))?;
    let keys = boxed_map.keys();
    let list = process.list_from_slice(&keys)?;

    Ok(list)
}
