// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Term;

use lumen_runtime_macros::native_implemented_function;

#[native_implemented_function(binary_to_list/1)]
pub fn native(process: &Process, binary: Term) -> exception::Result<Term> {
    let bytes = process.bytes_from_binary(binary)?;
    let byte_terms = bytes.iter().map(|byte| (*byte).into());

    process
        .list_from_iter(byte_terms)
        .map_err(|error| error.into())
}
