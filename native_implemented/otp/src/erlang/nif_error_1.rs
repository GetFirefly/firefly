use anyhow::*;

use firefly_rt::term::Term;

#[native_implemented::function(erlang:nif_error/1)]
pub fn result(reason: Term) -> Result<Term, NonNull<ErlangException>> {
    Err(error(
        reason,
        None,
        Trace::capture(),
        Some(anyhow!("explicit nif_error from Erlang").into()),
    )
    .into())
}
