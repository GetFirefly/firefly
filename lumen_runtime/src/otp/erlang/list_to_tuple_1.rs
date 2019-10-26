// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use lumen_runtime_macros::native_implemented_function;

#[native_implemented_function(list_to_tuple/1)]
pub fn native(process: &Process, list: Term) -> exception::Result {
    match list.decode().unwrap() {
        TypedTerm::Nil => process.tuple_from_slices(&[]).map_err(|error| error.into()),
        TypedTerm::List(cons) => {
            let vec: Vec<Term> = cons.into_iter().collect::<std::result::Result<_, _>>()?;

            process.tuple_from_slice(&vec).map_err(|error| error.into())
        }
        _ => Err(badarg!().into()),
    }
}
