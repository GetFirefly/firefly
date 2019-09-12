// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;

use hashbrown::HashMap;

use liblumen_alloc::badmap;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::{Boxed, Map, Term};

use lumen_runtime_macros::native_implemented_function;

#[native_implemented_function(merge/2)]
pub fn native(process: &Process, map1: Term, map2: Term) -> exception::Result {
    let result_map1: Result<Boxed<Map>, _> = map1.try_into();

    match result_map1 {
        Ok(map1) => {
            let result_map2: Result<Boxed<Map>, _> = map2.try_into();

            match result_map2 {
                Ok(map2) => {
                    let hash_map1: &HashMap<_, _> = map1.as_ref();
                    let hash_map2: &HashMap<_, _> = map2.as_ref();

                    let mut hash_map3: HashMap<Term, Term> =
                        HashMap::with_capacity(hash_map1.len() + hash_map2.len());

                    for (key, value) in hash_map1 {
                        hash_map3.insert(*key, *value);
                    }

                    for (key, value) in hash_map2 {
                        hash_map3.insert(*key, *value);
                    }

                    process
                        .map_from_hash_map(hash_map3)
                        .map_err(|error| error.into())
                }
                Err(_) => Err(badmap!(process, map2)),
            }
        }
        Err(_) => Err(badmap!(process, map1)),
    }
}
