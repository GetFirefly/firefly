use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::erlang::demonitor_2::demonitor;

#[native_implemented::function(erlang:demonitor/1)]
pub fn result(process: &Process, reference: Term) -> exception::Result<Term> {
    let reference_reference = term_try_into_local_reference!(reference)?;

    demonitor(process, &reference_reference, Default::default())
}
