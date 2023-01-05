#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use firefly_rt::term::Term;

/// `/=/2` infix operator.  Unlike `=/=`, converts between floats and integers.
#[native_implemented::function(erlang:/=/2)]
pub fn result(left: Term, right: Term) -> Term {
    left.ne(&right).into()
}
