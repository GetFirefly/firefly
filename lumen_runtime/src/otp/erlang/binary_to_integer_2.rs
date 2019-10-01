// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;

use num_bigint::BigInt;

use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::alloc::heap_alloc::HeapAlloc;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::Term;

use lumen_runtime_macros::native_implemented_function;

use crate::otp::erlang::base::Base;

#[native_implemented_function(binary_to_integer/2)]
pub fn native<'process>(process: &'process Process, binary: Term, base: Term) -> exception::Result {
    let mut heap = process.acquire_heap();
    let s: &str = heap.str_from_binary(binary)?;
    let base: Base = base.try_into()?;

    let bytes = s.as_bytes();

    match BigInt::parse_bytes(bytes, base.radix()) {
        Some(big_int) => {
            let term = heap.integer(big_int)?;

            Ok(term)
        }
        None => Err(badarg!().into()),
    }
}
