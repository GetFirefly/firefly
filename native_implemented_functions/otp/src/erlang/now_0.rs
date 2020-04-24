use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use native_implemented_function::native_implemented_function;

use crate::erlang::timestamp_0;

// now/0 is deprecated. We are implementing here using timestamp/0
// which is not deprecated.
#[native_implemented_function(now/0)]
pub fn result(process: &Process) -> exception::Result<Term> {
    timestamp_0::result(process)
}
