// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::alloc::heap_alloc::HeapAlloc;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use lumen_runtime_macros::native_implemented_function;

#[native_implemented_function(tuple_to_list/1)]
pub fn native(process: &Process, tuple: Term) -> exception::Result {
    let tuple: Boxed<Tuple> = tuple.try_into()?;
    let mut heap = process.acquire_heap();
    let mut acc = Term::NIL;

    for element in tuple.iter().rev() {
        acc = heap.cons(element, acc)?;
    }

    Ok(acc)
}
