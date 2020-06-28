#[cfg(all(not(any(target_arch = "wasm32", feature = "runtime_minimal")), test))]
mod test;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::alloc::TermAlloc;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

#[native_implemented::function(tuple_to_list/1)]
pub fn result(process: &Process, tuple: Term) -> exception::Result<Term> {
    let tuple = term_try_into_tuple!(tuple)?;
    let mut heap = process.acquire_heap();
    let mut acc = Term::NIL;

    for element in tuple.iter().rev() {
        acc = heap.cons(*element, acc)?.into();
    }

    Ok(acc)
}
