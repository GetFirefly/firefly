#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::ptr::NonNull;

use firefly_rt::error::ErlangException;
use firefly_rt::term::Term;

use crate::runtime::context::*;

/// `not/1` prefix operator.
#[native_implemented::function(erlang:not/1)]
pub fn result(boolean: Term) -> Result<Term, NonNull<ErlangException>> {
    let boolean_bool: bool = term_try_into_bool("boolean", boolean)?;
    let output = !boolean_bool;

    Ok(output.into())
}
