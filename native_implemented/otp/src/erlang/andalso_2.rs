use std::convert::TryInto;

use anyhow::*;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::term::prelude::Term;

/// `andalso/2` infix operator.
///
/// Short-circuiting, but doesn't enforce `right` is boolean.  If you need to enforce `boolean` for
/// both operands, use `and_2`.
#[native_implemented::function(erlang:andalso/2)]
fn result(boolean: Term, term: Term) -> exception::Result<Term> {
    let boolean_bool: bool = boolean.try_into().context("left must be a bool")?;

    if boolean_bool {
        Ok(term)
    } else {
        // always `false.into()`, but this is faster
        Ok(boolean)
    }
}
