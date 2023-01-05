use std::ptr::NonNull;

use firefly_rt::error::ErlangException;
use firefly_rt::term::Term;

/// `and/2` infix operator.
///
/// **NOTE: NOT SHORT-CIRCUITING!**  Use `andalso/2` for short-circuiting, but it doesn't enforce
/// that `right` is boolean.
#[native_implemented::function(erlang:and/2)]
fn result(left_boolean: Term, right_boolean: Term) -> Result<Term, NonNull<ErlangException>> {
    boolean_infix_operator!(left_boolean, right_boolean, &)
}
