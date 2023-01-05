use std::ptr::NonNull;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::Term;

use crate::erlang::demonitor_2::demonitor;

#[native_implemented::function(erlang:demonitor/1)]
pub fn result(process: &Process, reference: Term) -> Result<Term, NonNull<ErlangException>> {
    let reference_reference = term_try_into_local_reference!(reference)?;

    demonitor(process, &reference_reference, Default::default())
}
