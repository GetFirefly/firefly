#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use anyhow::*;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::trace::Trace;
use liblumen_alloc::erts::term::prelude::Term;
use liblumen_alloc::exit;

#[native_implemented::function(erlang:exit/1)]
fn result(reason: Term) -> exception::Result<Term> {
    Err(exit!(
        reason,
        Trace::capture(),
        anyhow!("explicit exit from Erlang").into()
    )
    .into())
}
