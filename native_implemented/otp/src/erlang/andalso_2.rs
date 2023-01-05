use std::convert::TryInto;
use std::ptr::NonNull;

use anyhow::*;

use firefly_rt::error::ErlangException;
use firefly_rt::term::Term;

/// `andalso/2` infix operator.
///
/// Short-circuiting, but doesn't enforce `right` is boolean.  If you need to enforce `boolean` for
/// both operands, use `and_2`.
#[native_implemented::function(erlang:andalso/2)]
fn result(boolean: Term, term: Term) -> Result<Term, NonNull<ErlangException>> {
    let boolean_bool: bool = boolean.try_into().context("left must be a bool")?;

    if boolean_bool {
        Ok(term)
    } else {
        // always `false.into()`, but this is faster
        Ok(boolean)
    }
}
