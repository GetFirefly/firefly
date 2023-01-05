#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::ptr::NonNull;

use anyhow::*;

use firefly_rt::error::ErlangException;
use firefly_rt::term::Term;

#[native_implemented::function(erlang:+/1)]
pub fn result(number: Term) -> Result<Term, NonNull<ErlangException>> {
    if number.is_number() {
        Ok(number)
    } else {
        Err(badarith(
            Trace::capture(),
            Some(anyhow!("number ({}) is not an integer or a float", number).into()),
        )
        .into())
    }
}
