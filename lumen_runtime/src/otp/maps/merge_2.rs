// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;

use hashbrown::HashMap;

use liblumen_alloc::badmap;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::erts::exception;

use lumen_runtime_macros::native_implemented_function;

#[native_implemented_function(merge/2)]
pub fn native(process: &Process, map1: Term, map2: Term) -> exception::Result<Term> {
    let result_map1: Result<Boxed<Map>, _> = map1.try_into();

    match result_map1 {
        Ok(map1) => {
            let result_map2: Result<Boxed<Map>, _> = map2.try_into();

            match result_map2 {
                Ok(map2) => {
                    let mut merged: HashMap<Term, Term> =
                        HashMap::with_capacity(map1.len() + map2.len());

                    for (key, value) in map1.iter() {
                        merged.insert(*key, *value);
                    }

                    for (key, value) in map2.iter() {
                        merged.insert(*key, *value);
                    }

                    process
                        .map_from_hash_map(merged)
                        .map_err(|error| error.into())
                }
                Err(_) => Err(badmap!(process, map2)),
            }
        }
        Err(_) => Err(badmap!(process, map1)),
    }
}
