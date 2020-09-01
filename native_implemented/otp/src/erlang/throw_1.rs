#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use anyhow::*;

use liblumen_alloc::erts::exception::{self, *};
use liblumen_alloc::erts::process::trace::Trace;
use liblumen_alloc::erts::term::prelude::Term;

#[native_implemented::function(erlang:throw/1)]
pub fn result(reason: Term) -> exception::Result<Term> {
    Err(throw(
        reason,
        Trace::capture(),
        Some(anyhow!("explicit throw from Erlang").into()),
    )
    .into())
}
