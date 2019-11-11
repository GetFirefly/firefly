// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::Term;
use lumen_runtime_macros::native_implemented_function;

use crate::otp;

/// Returns the size, in bytes, of the binary that would be result from iolist_to_binary/1
#[native_implemented_function(iolist_size/1)]
pub fn native(process: &Process, iolist_or_binary: Term) -> exception::Result {
    let bin = otp::erlang::iolist_to_binary_1::native(process, iolist_or_binary)?;
    let bytes = process.bytes_from_binary(bin)?;
    Ok(process.integer(bytes.len()).unwrap())
}
