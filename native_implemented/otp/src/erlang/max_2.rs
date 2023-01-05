#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use firefly_rt::term::Term;

/// `max/2`
///
/// Returns the largest of `Term1` and `Term2`. If the terms are equal, `Term1` is returned.
#[native_implemented::function(erlang:max/2)]
pub fn result(term1: Term, term2: Term) -> Term {
    // Flip the order because for Rust `max` returns the second argument when equal, but Erlang
    // returns the first
    term2.max(term1)
}
