// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;

use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use lumen_runtime_macros::native_implemented_function;

use crate::binary::ToTermOptions;

#[native_implemented_function(binary_to_term/2)]
pub fn native(_process: &Process, binary: Term, options: Term) -> exception::Result {
    let _to_term_options: ToTermOptions = options.try_into()?;

    match binary.decode().unwrap() {
        TypedTerm::HeapBinary(_heap_binary) => unimplemented!(),
        TypedTerm::ProcBin(_process_binary) => unimplemented!(),
        TypedTerm::SubBinary(_subbinary) => unimplemented!(),
        _ => Err(badarg!().into()),
    }
}
