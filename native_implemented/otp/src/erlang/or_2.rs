#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use firefly_rt::error::ErlangException;
use firefly_rt::term::Term;

/// `or/2` infix operator.
///
/// **NOTE: NOT SHORT-CIRCUITING!**
#[native_implemented::function(erlang:or/2)]
pub fn result(left_boolean: Term, right_boolean: Term) -> Result<Term, NonNull<ErlangException>> {
    boolean_infix_operator!(left_boolean, right_boolean, |)
}
