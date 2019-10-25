// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;

use liblumen_alloc::badmap;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use lumen_runtime_macros::native_implemented_function;

#[native_implemented_function(values/1)]
pub fn native(process: &Process, map: Term) -> exception::Result {
    let result_map: Result<Boxed<Map>, _> = map.try_into();

    match result_map {
        Ok(map) => {
            let values = map.values();
            let list = process.list_from_slice(&values)?;

            Ok(list)
        }

        Err(_) => Err(badmap!(process, map)),
    }
}
