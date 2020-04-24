// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use liblumen_alloc::erts::term::prelude::Term;

use native_implemented_function::native_implemented_function;

/// `max/2`
///
/// Returns the largest of `Term1` and `Term2`. If the terms are equal, `Term1` is returned.
#[native_implemented_function(max/2)]
pub fn result(term1: Term, term2: Term) -> Term {
    // Flip the order because for Rust `max` returns the second argument when equal, but Erlang
    // returns the first
    term2.max(term1)
}
