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

#[native_implemented_function(put/3)]
pub fn result(process: &Process, key: Term, value: Term, map: Term) -> exception::Result<Term> {
    let boxed_map = term_try_into_map_or_badmap!(process, map)?;

    match boxed_map.put(key, value) {
        Some(hash_map) => Ok(process.map_from_hash_map(hash_map)?),
        None => Ok(map),
    }
}
