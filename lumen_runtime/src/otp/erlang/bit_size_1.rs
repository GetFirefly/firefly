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

#[native_implemented_function(bit_size/1)]
pub fn native(process: &Process, bitstring: Term) -> exception::Result<Term> {
    let option_total_bit_len = match bitstring.decode().unwrap() {
        TypedTerm::HeapBinary(heap_binary) => Some(heap_binary.total_bit_len()),
        TypedTerm::ProcBin(process_binary) => Some(process_binary.total_bit_len()),
        TypedTerm::SubBinary(subbinary) => Some(subbinary.total_bit_len()),
        _ => None,
    };

    match option_total_bit_len {
        Some(total_bit_len) => Ok(process.integer(total_bit_len)?),
        None => Err(badarg!().into()),
    }
}
