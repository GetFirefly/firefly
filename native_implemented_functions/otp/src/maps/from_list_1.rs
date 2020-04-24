// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use native_implemented_function::native_implemented_function;

#[native_implemented_function(from_list/1)]
pub fn result(process: &Process, list: Term) -> exception::Result<Term> {
    let hash_map = Map::from_list(list)?;
    let map = process.map_from_hash_map(hash_map)?;

    Ok(map)
}
