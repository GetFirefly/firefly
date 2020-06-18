#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::term::prelude::Term;

use crate::runtime::context::*;

/// `not/1` prefix operator.
#[native_implemented::function(erlang:not/1)]
pub fn result(boolean: Term) -> exception::Result<Term> {
    let boolean_bool: bool = term_try_into_bool("boolean", boolean)?;
    let output = !boolean_bool;

    Ok(output.into())
}
