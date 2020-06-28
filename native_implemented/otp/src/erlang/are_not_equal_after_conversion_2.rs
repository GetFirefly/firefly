#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use liblumen_alloc::erts::term::prelude::Term;

/// `/=/2` infix operator.  Unlike `=/=`, converts between floats and integers.
#[native_implemented::function(/=/2)]
pub fn result(left: Term, right: Term) -> Term {
    left.ne(&right).into()
}
