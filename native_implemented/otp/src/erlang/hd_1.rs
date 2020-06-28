#[cfg(all(not(any(target_arch = "wasm32", feature = "runtime_minimal")), test))]
mod test;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::term::prelude::*;

#[native_implemented::function(hd/1)]
pub fn result(list: Term) -> exception::Result<Term> {
    let cons: Boxed<Cons> = term_try_into_non_empty_list!(list)?;

    Ok(cons.head)
}
