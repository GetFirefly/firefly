#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use liblumen_alloc::erts::term::prelude::Term;

/// `min/2`
///
/// Returns the smallest of `Term1` and `Term2`. If the terms are equal, `Term1` is returned.
#[native_implemented::function(min/2)]
pub fn result(term1: Term, term2: Term) -> Term {
    term1.min(term2)
}
