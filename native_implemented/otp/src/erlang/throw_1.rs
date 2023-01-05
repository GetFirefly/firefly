#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::ptr::NonNull;
use anyhow::*;

use firefly_rt::backtrace::Trace;
use firefly_rt::error::ErlangException;

use firefly_rt::term::Term;

#[native_implemented::function(erlang:throw/1)]
pub fn result(reason: Term) -> Result<Term, NonNull<ErlangException>> {
    Err(throw(
        reason,
        Trace::capture(),
        Some(anyhow!("explicit throw from Erlang").into()),
    )
    .into())
}
