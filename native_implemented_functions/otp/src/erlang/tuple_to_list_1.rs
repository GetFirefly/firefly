// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::alloc::TermAlloc;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use native_implemented_function::native_implemented_function;

#[native_implemented_function(tuple_to_list/1)]
pub fn result(process: &Process, tuple: Term) -> exception::Result<Term> {
    let tuple = term_try_into_tuple!(tuple)?;
    let mut heap = process.acquire_heap();
    let mut acc = Term::NIL;

    for element in tuple.iter().rev() {
        acc = heap.cons(*element, acc)?.into();
    }

    Ok(acc)
}
