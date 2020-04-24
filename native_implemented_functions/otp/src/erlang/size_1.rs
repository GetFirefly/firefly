// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use anyhow::*;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use native_implemented_function::native_implemented_function;

#[native_implemented_function(size/1)]
pub fn result(process: &Process, binary_or_tuple: Term) -> exception::Result<Term> {
    let option_size = match binary_or_tuple.decode().unwrap() {
        TypedTerm::Tuple(tuple) => Some(tuple.len()),
        TypedTerm::HeapBinary(heap_binary) => Some(heap_binary.full_byte_len()),
        TypedTerm::ProcBin(process_binary) => Some(process_binary.full_byte_len()),
        TypedTerm::SubBinary(subbinary) => Some(subbinary.full_byte_len()),
        _ => None,
    };

    match option_size {
        Some(size) => Ok(process.integer(size)?),
        None => Err(TypeError)
            .context(format!(
                "binary_or_tuple ({}) is neither a binary nor a tuple",
                binary_or_tuple
            ))
            .map_err(From::from),
    }
}
