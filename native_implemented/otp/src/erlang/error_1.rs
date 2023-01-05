use std::ptr::NonNull;

use anyhow::*;

use firefly_rt::backtrace::Trace;
use firefly_rt::error::ErlangException;
use firefly_rt::term::Term;

#[native_implemented::function(erlang:error/1)]
pub fn result(reason: Term) -> Result<Term, NonNull<ErlangException>> {
    Err(error(
        reason,
        None,
        Trace::capture(),
        Some(anyhow!("explicit error from Erlang").into()),
    )
    .into())
}
