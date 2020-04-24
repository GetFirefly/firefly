// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::term::prelude::Term;

use native_implemented_function::native_implemented_function;

/// `and/2` infix operator.
///
/// **NOTE: NOT SHORT-CIRCUITING!**  Use `andalso/2` for short-circuiting, but it doesn't enforce
/// that `right` is boolean.
#[native_implemented_function(and/2)]
fn result(left_boolean: Term, right_boolean: Term) -> exception::Result<Term> {
    boolean_infix_operator!(left_boolean, right_boolean, &)
}
