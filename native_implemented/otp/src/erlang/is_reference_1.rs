#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use liblumen_alloc::erts::term::prelude::*;

#[native_implemented::function(erlang:is_reference/1)]
pub fn result(term: Term) -> Term {
    match term.decode().unwrap() {
        TypedTerm::Reference(_) => true,
        TypedTerm::ExternalReference(_) => true,
        TypedTerm::ResourceReference(_) => true,
        _ => false,
    }
    .into()
}
