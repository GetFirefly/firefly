// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use lumen_runtime_macros::native_implemented_function;

#[native_implemented_function(append_element/2)]
pub fn native(process: &Process, tuple: Term, element: Term) -> exception::Result {
    let internal: Boxed<Tuple> = tuple.try_into()?;
    let new_tuple = process.tuple_from_slices(&[&internal[..], &[element]])?;

    Ok(new_tuple)
}
