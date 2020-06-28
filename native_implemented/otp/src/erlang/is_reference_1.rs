#[cfg(all(not(any(target_arch = "wasm32", feature = "runtime_minimal")), test))]
mod test;

use liblumen_alloc::erts::term::prelude::*;

#[native_implemented::function(is_reference/1)]
pub fn result(term: Term) -> Term {
    match term.decode().unwrap() {
        TypedTerm::Reference(_) => true,
        TypedTerm::ExternalReference(_) => true,
        TypedTerm::ResourceReference(_) => true,
        _ => false,
    }
    .into()
}
