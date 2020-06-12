// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use liblumen_alloc::erts::term::prelude::*;

use native_implemented_function::native_implemented_function;

#[native_implemented_function(is_reference/1)]
pub fn result(term: Term) -> Term {
    match term.decode().unwrap() {
        TypedTerm::Reference(_) => true,
        TypedTerm::ExternalReference(_) => true,
        TypedTerm::ResourceReference(_) => true,
        _ => false,
    }
    .into()
}
