#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use anyhow::*;

use liblumen_alloc::error_with_source;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::term::prelude::Term;

#[native_implemented::function(erlang:error/2)]
pub fn result(reason: Term, arguments: Term) -> exception::Result<Term> {
    Err(error_with_source!(
        reason,
        arguments,
        anyhow!("explicit error from Erlang").into()
    )
    .into())
}
