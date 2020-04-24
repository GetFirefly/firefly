// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use liblumen_alloc::erts::term::prelude::Term;

use native_implemented_function::native_implemented_function;

/// `/=/2` infix operator.  Unlike `=/=`, converts between floats and integers.
#[native_implemented_function(/=/2)]
pub fn result(left: Term, right: Term) -> Term {
    left.ne(&right).into()
}
