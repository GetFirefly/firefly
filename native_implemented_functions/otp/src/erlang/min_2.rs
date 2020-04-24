// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use liblumen_alloc::erts::term::prelude::Term;

use native_implemented_function::native_implemented_function;

/// `min/2`
///
/// Returns the smallest of `Term1` and `Term2`. If the terms are equal, `Term1` is returned.
#[native_implemented_function(min/2)]
pub fn result(term1: Term, term2: Term) -> Term {
    term1.min(term2)
}
