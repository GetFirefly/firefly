use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use lumen_runtime_macros::native_implemented_function;

use crate::erlang::timestamp_0;

// now/0 is deprecated. We are implementing here using timestamp/0
// which is not deprecated.
#[native_implemented_function(now/0)]
pub fn native(process: &Process) -> exception::Result<Term> {
    timestamp_0::native(process)
}
