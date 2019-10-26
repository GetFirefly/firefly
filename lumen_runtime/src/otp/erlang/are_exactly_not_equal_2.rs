// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use liblumen_alloc::erts::term::prelude::Term;

use lumen_runtime_macros::native_implemented_function;

/// `=/=/2` infix operator.  Unlike `!=`, does not convert between floats and integers.
#[native_implemented_function(=/=/2)]
pub fn native(left: Term, right: Term) -> Term {
    let left = left.decode().unwrap();
    let right = right.decode().unwrap();
    left.exactly_ne(&right).into()
}
