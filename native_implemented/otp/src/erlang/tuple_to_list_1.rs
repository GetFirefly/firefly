#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::ptr::NonNull;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::Term;

#[native_implemented::function(erlang:tuple_to_list/1)]
pub fn result(process: &Process, tuple: Term) -> Result<Term, NonNull<ErlangException>> {
    let tuple = term_try_into_tuple!(tuple)?;
    let mut heap = process.acquire_heap();
    let mut acc = Term::Nil;

    for element in tuple.iter().rev() {
        acc = heap.cons(*element, acc)?.into();
    }

    Ok(acc)
}
