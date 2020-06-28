#[cfg(all(not(any(target_arch = "wasm32", feature = "runtime_minimal")), test))]
mod test;

use liblumen_alloc::erts::term::prelude::*;

#[native_implemented::function(is_tuple/1)]
pub fn result(term: Term) -> Term {
    match term.decode() {
        Ok(TypedTerm::Tuple(_)) => true.into(),
        _ => false.into(),
    }
}
