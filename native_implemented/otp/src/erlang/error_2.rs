#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use anyhow::*;

use liblumen_alloc::error;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::term::prelude::Term;

#[native_implemented::function(error/2)]
pub fn result(reason: Term, arguments: Term) -> exception::Result<Term> {
    Err(error!(
        reason,
        arguments,
        anyhow!("explicit error from Erlang").into()
    )
    .into())
}
