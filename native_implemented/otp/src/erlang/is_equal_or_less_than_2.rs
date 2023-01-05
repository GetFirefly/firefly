#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use firefly_rt::term::Term;

/// `=</2` infix operator.  Floats and integers are converted.
///
/// **NOTE: `=</2` is not a typo.  Unlike `>=/2`, which has the `=` second, Erlang put the `=` first
/// for `=</2`, instead of the more common `<=`.
#[native_implemented::function(erlang:=</2)]
pub fn result(left: Term, right: Term) -> Term {
    left.le(&right).into()
}
