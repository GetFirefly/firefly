#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use anyhow::*;

use liblumen_alloc::erts::exception::{self, error};
use liblumen_alloc::erts::process::trace::Trace;
use liblumen_alloc::erts::term::prelude::Term;

#[native_implemented::function(erlang:error/1)]
pub fn result(reason: Term) -> exception::Result<Term> {
    Err(error(
        reason,
        None,
        Trace::capture(),
        Some(anyhow!("explicit error from Erlang").into()),
    )
    .into())
}
