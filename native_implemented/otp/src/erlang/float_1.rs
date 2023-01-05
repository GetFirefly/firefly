use std::convert::TryInto;
use std::ptr::NonNull;

use anyhow::*;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::Term;

#[native_implemented::function(erlang:float/1)]
pub fn result(process: &Process, number: Term) -> Result<Term, NonNull<ErlangException>> {
    if number.is_float() {
        Ok(number)
    } else {
        let f: f64 = number
            .try_into()
            .with_context(|| term_is_not_number!(number))?;

        Ok(f.into())
    }
}
