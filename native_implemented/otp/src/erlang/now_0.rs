use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::erlang::timestamp_0;

// now/0 is deprecated. We are implementing here using timestamp/0
// which is not deprecated.
#[native_implemented::function(erlang:now/0)]
pub fn result(process: &Process) -> exception::Result<Term> {
    timestamp_0::result(process)
}
