// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::num::FpCategory;

use liblumen_core::locks::MutexGuard;

use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::alloc::heap_alloc::HeapAlloc;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::Term;

use lumen_runtime_macros::native_implemented_function;

#[native_implemented_function(binary_to_float/1)]
pub fn native<'process>(process: &'process Process, binary: Term) -> exception::Result {
    let mut heap: MutexGuard<'process, _> = process.acquire_heap();
    let s: &str = heap.str_from_binary(binary)?;

    match s.parse::<f64>() {
        Ok(inner) => {
            match inner.classify() {
                FpCategory::Normal | FpCategory::Subnormal =>
                // unlike Rust, Erlang requires float strings to have a decimal point
                {
                    if (inner.fract() == 0.0) & !s.chars().any(|b| b == '.') {
                        Err(badarg!().into())
                    } else {
                        heap.float(inner).map_err(|error| error.into())
                    }
                }
                // Erlang has no support for Nan, +inf or -inf
                FpCategory::Nan | FpCategory::Infinite => Err(badarg!().into()),
                FpCategory::Zero => {
                    // Erlang does not track the difference without +0 and -0.
                    let zero = inner.abs();

                    heap.float(zero).map_err(|error| error.into())
                }
            }
        }
        Err(_) => Err(badarg!().into()),
    }
}
