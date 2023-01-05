use firefly_rt::process::Process;
use firefly_rt::term::Term;

use crate::erlang::timestamp_0;

// now/0 is deprecated. We are implementing here using timestamp/0
// which is not deprecated.
#[native_implemented::function(erlang:now/0)]
pub fn result(process: &Process) -> Term {
    timestamp_0::result(process)
}
