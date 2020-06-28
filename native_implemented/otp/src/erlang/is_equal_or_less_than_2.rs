#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use liblumen_alloc::erts::term::prelude::Term;

/// `=</2` infix operator.  Floats and integers are converted.
///
/// **NOTE: `=</2` is not a typo.  Unlike `>=/2`, which has the `=` second, Erlang put the `=` first
/// for `=</2`, instead of the more common `<=`.
#[native_implemented::function(=</2)]
pub fn result(left: Term, right: Term) -> Term {
    left.le(&right).into()
}
