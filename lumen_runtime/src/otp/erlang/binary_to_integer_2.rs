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

#[native_implemented_function(binary_to_integer/2)]
pub fn native<'process>(process: &'process Process, binary: Term, base: Term) -> exception::Result {
    let mut heap = process.acquire_heap();
    let s: &str = heap.str_from_binary(binary)?;
    let radix: usize = base.try_into()?;

    if 2 <= radix && radix <= 36 {
        let bytes = s.as_bytes();

        match BigInt::parse_bytes(bytes, radix as u32) {
            Some(big_int) => {
                let term = heap.integer(big_int)?;

                Ok(term)
            }
            None => Err(badarg!()),
        }
    } else {
        Err(badarg!())
    }
    .map_err(|error| error.into())
}
